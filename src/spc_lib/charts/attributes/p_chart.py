import numpy as np
from spc_lib.core.base_chart import BaseControlChart


class PChart(BaseControlChart):
    """p-chart for the proportion of nonconforming units."""

    def __init__(self, data, n_fixed=None, datetimes=None, target=None, usl=None, lsl=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.n_fixed = n_fixed
        self.main_label = "p-chart: Proportion of nonconforming units"
        self.disp_label = None
        self.p_values = None
        self.n_values = None
        self.p_bar = None
        self.n_bar = None

    def fit(self, baseline_mask=None, method='classic', use_average_n=False):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        if self.n_fixed is not None:
            self.p_values = np.asarray(self.data).flatten()
            self.n_values = np.full(len(self.p_values), self.n_fixed)
        elif self.data.ndim == 2 and self.data.shape[1] == 2:
            self.p_values = self.data[:, 0] / self.data[:, 1]
            self.n_values = self.data[:, 1]
        else:
            raise ValueError("For variable sample size, data must be a 2D array with shape (n_samples, 2): [x, n]")

        base_p = self.p_values[baseline_mask]
        base_n = self.n_values[baseline_mask]
        if self.target is not None:
            self.p_bar = self.target
        else:
            total_n = np.sum(base_n)
            self.p_bar = np.sum(base_p * base_n) / total_n if total_n > 0 else np.mean(base_p)
        self.n_bar = np.mean(base_n)

        if method == 'classic':
            self.cl_main = self.p_bar
            if self.n_fixed is not None or use_average_n:
                n_eval = self.n_fixed if self.n_fixed is not None else self.n_bar
                sigma_p = np.sqrt(self.p_bar * (1 - self.p_bar) / n_eval)
                self.ucl_main = self.p_bar + 3 * sigma_p
                self.lcl_main = max(0, self.p_bar - 3 * sigma_p)
            else:
                self.ucl_main = np.zeros(len(self.p_values))
                self.lcl_main = np.zeros(len(self.p_values))
                for i, n_i in enumerate(self.n_values):
                    sigma_p_i = np.sqrt(self.p_bar * (1 - self.p_bar) / n_i)
                    self.ucl_main[i] = self.p_bar + 3 * sigma_p_i
                    self.lcl_main[i] = max(0, self.p_bar - 3 * sigma_p_i)
            self.sigma_est = np.sqrt(self.p_bar * (1 - self.p_bar))
        elif method == 'percentiles':
            self.cl_main = np.median(base_p)
            self.ucl_main = np.percentile(base_p, 99.865)
            self.lcl_main = np.percentile(base_p, 0.135)
            self.sigma_est = np.std(base_p, ddof=1)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.stat_main = self.p_values
        self.cl_disp = self.ucl_disp = self.lcl_disp = None
        return self
