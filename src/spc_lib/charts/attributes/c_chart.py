import numpy as np
from spc_lib.core.base_chart import BaseControlChart


class CChart(BaseControlChart):
    """c-chart for the number of defects."""

    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.main_label = "c-chart: Number of defects"
        self.disp_label = None
        self.c_values = None
        self.c_bar = None

    def fit(self, baseline_mask=None, method='classic'):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        self.c_values = np.asarray(self.data).flatten()
        base_c = self.c_values[baseline_mask]
        self.c_bar = self.target if self.target is not None else np.mean(base_c)

        if method == 'classic':
            self.cl_main = self.c_bar
            sigma_c = np.sqrt(self.c_bar)
            self.ucl_main = self.c_bar + 3 * sigma_c
            self.lcl_main = max(0, self.c_bar - 3 * sigma_c)
            self.sigma_est = sigma_c
        elif method == 'percentiles':
            self.cl_main = np.median(base_c)
            self.ucl_main = np.percentile(base_c, 99.865)
            self.lcl_main = np.percentile(base_c, 0.135)
            self.sigma_est = np.std(base_c, ddof=1)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.stat_main = self.c_values
        self.cl_disp = self.ucl_disp = self.lcl_disp = None
        return self
