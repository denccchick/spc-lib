import numpy as np
import warnings
from spc_lib.core.base_chart import BaseControlChart


class CUSUMVarianceChart(BaseControlChart):
    """CUSUM chart for monitoring process variance."""

    def __init__(self, data, datetimes=None, target_mean=None, target_std=None,
                 usl=None, lsl=None, h=5.0, k=0.5):
        super().__init__(data, datetimes, None, usl, lsl)
        self.target_mean, self.target_std = target_mean, target_std
        self.h, self.k = h, k
        self.main_label = "CUSUM for variance (upper/lower)"
        self.disp_label = None
        self.cusum_upper = self.cusum_lower = self.v_values = self.sigma_est = None

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
        mu = self.target_mean if self.target_mean is not None else (np.mean(base_x) if method == 'classic' else np.median(base_x))
        if self.target_std is not None:
            sigma = self.target_std
        elif method == 'classic' and n > 1:
            r = np.ptp(self.data[baseline_mask], axis=1)
            d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
                  7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
            sigma = np.mean(r) / d2.get(n, 1.128)
        else:
            sigma = np.std(base_x, ddof=1)
        if sigma > 0:
            y = (x - mu) / sigma
        else:
            y = x - mu
            warnings.warn("Sigma is zero, using unnormalized statistic", UserWarning)
        self.v_values = (np.sqrt(np.abs(y)) - 0.822) / 0.349
        self.cusum_upper = np.zeros(len(x))
        self.cusum_lower = np.zeros(len(x))
        if method == 'classic':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i - 1] + self.v_values[i] - self.k)
                self.cusum_lower[i] = max(0, self.cusum_lower[i - 1] - self.k - self.v_values[i])
            self.cl_main, self.ucl_main, self.lcl_main = 0, self.h, self.h
        elif method == 'percentiles':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i - 1] + self.v_values[i])
                self.cusum_lower[i] = max(0, self.cusum_lower[i - 1] - self.v_values[i])
            self.cl_main = 0
            self.ucl_main = np.percentile(self.cusum_upper[baseline_mask], 99.865)
            self.lcl_main = np.percentile(self.cusum_lower[baseline_mask], 99.865)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.target, self.target_mean = mu, mu
        self.target_std, self.sigma_est, self.stat_main = sigma, sigma, x
        return self

    def plot(self, start=None, end=None, last_n=30, show_spec=False):
        """Plot CUSUM chart for variance."""
        from spc_lib.visualization.plotly_engine import _plot_cusum_chart
        return _plot_cusum_chart(self, start, end, last_n, show_spec)
