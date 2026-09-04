import numpy as np
import warnings
from spc_lib.core.base_chart import BaseControlChart


class CUSUMChart(BaseControlChart):
    """CUSUM chart for detecting small process shifts."""

    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None,
                 h=5.0, k=0.5, std_est=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.h, self.k, self.std_est = h, k, std_est
        self.main_label = "CUSUM: Cumulative sum (upper/lower)"
        self.disp_label = None
        self.cusum_upper = self.cusum_lower = self.sigma_est = None

    def fit(self, baseline_mask=None, method='classic'):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)
        if self.data.ndim == 2:
            if self.subgroup_size == 1:
                x, n = self.data.flatten(), 1
            else:
                x, n = np.mean(self.data, axis=1), self.subgroup_size
        else:
            warnings.warn("1D array provided. It is recommended to use 2D format: data.reshape(-1, 1)", UserWarning)
            x, n = self.data, 1
        base_x = x[baseline_mask]
        if self.std_est is None:
            if n > 1 and method == 'classic':
                r = np.ptp(self.data[baseline_mask], axis=1)
                d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
                      7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
                sigma_est = np.mean(r) / d2.get(n, 1.128)
            else:
                sigma_est = np.std(base_x, ddof=1)
        else:
            sigma_est = self.std_est
        target = self.target if self.target is not None else (np.mean(base_x) if method == 'classic' else np.median(base_x))
        if sigma_est > 0:
            z = (x - target) / sigma_est
        else:
            z = x - target
            warnings.warn("Sigma is zero, using unnormalized statistic", UserWarning)
        self.cusum_upper = np.zeros(len(x))
        self.cusum_lower = np.zeros(len(x))
        if method == 'classic':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i - 1] + z[i] - self.k)
                self.cusum_lower[i] = max(0, self.cusum_lower[i - 1] - z[i] - self.k)
            self.cl_main, self.ucl_main, self.lcl_main = 0, self.h, self.h
        elif method == 'percentiles':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i - 1] + z[i])
                self.cusum_lower[i] = max(0, self.cusum_lower[i - 1] - z[i])
            self.cl_main = 0
            self.ucl_main = np.percentile(self.cusum_upper[baseline_mask], 99.865)
            self.lcl_main = np.percentile(self.cusum_lower[baseline_mask], 99.865)
        elif method == 'made':
            mad = np.median(np.abs(base_x - np.median(base_x)))
            robust_sigma = 1.4826 * mad if mad > 0 else 1e-10
            z_robust = (x - target) / robust_sigma
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i - 1] + z_robust[i] - self.k)
                self.cusum_lower[i] = max(0, self.cusum_lower[i - 1] - z_robust[i] - self.k)
            self.cl_main, self.ucl_main, self.lcl_main = 0, self.h, self.h
            sigma_est = robust_sigma
        else:
            raise ValueError(f"Unknown method: {method}")
        self.target, self.sigma_est, self.stat_main = target, sigma_est, x
        return self

    def get_cusum_stats(self):
        return self.cusum_upper, self.cusum_lower
