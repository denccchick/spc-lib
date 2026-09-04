import numpy as np
import warnings
from spc_lib.core.base_chart import BaseControlChart


class EWMAChart(BaseControlChart):
    """EWMA chart for detecting small process shifts."""

    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None,
                 lambda_=0.2, L=3, std_est=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.lambda_, self.L, self.std_est = lambda_, L, std_est
        self.main_label = "EWMA: Exponentially Weighted Moving Average"
        self.disp_label = None
        self.ewma_values = self.ewma_sigma = self.sigma_est = None

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
        self.ewma_values = np.zeros(len(x))
        self.ewma_values[0] = target
        for i in range(1, len(x)):
            self.ewma_values[i] = self.lambda_ * x[i] + (1 - self.lambda_) * self.ewma_values[i - 1]
        weights = np.array([np.sqrt(self.lambda_ / (2 - self.lambda_) * (1 - (1 - self.lambda_) ** (2 * i))) for i in range(1, len(x) + 1)])
        self.ewma_sigma = sigma_est * weights
        if method == 'classic':
            self.cl_main = target
            self.ucl_main = target + self.L * self.ewma_sigma
            self.lcl_main = target - self.L * self.ewma_sigma
        elif method == 'percentiles':
            base_ewma = self.ewma_values[baseline_mask]
            self.cl_main = np.median(base_ewma)
            self.ucl_main = np.percentile(base_ewma, 99.865)
            self.lcl_main = np.percentile(base_ewma, 0.135)
        elif method == 'made':
            mad = np.median(np.abs(base_x - np.median(base_x)))
            sigma_est = 1.4826 * mad if mad > 0 else 1e-10
            self.ewma_sigma = sigma_est * weights
            self.cl_main = target
            self.ucl_main = target + self.L * self.ewma_sigma
            self.lcl_main = target - self.L * self.ewma_sigma
        else:
            raise ValueError(f"Unknown method: {method}")
        self.target, self.sigma_est, self.stat_main = target, sigma_est, self.ewma_values
        return self

    def get_ewma_values(self):
        return self.ewma_values, self.ewma_sigma
