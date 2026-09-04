import numpy as np
from spc_lib.core.base_chart import BaseControlChart
from .constants import SPC_CONSTANTS


class XBarSChart(BaseControlChart):
    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.main_label = "X-bar: Subgroup means"
        self.disp_label = "S: Subgroup standard deviations"

    def fit(self, baseline_mask=None, method='classic'):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)
        n = self.subgroup_size
        self.stat_main = np.mean(self.data, axis=1)
        self.stat_disp = np.std(self.data, axis=1, ddof=1)
        base_xbar, base_s = self.stat_main[baseline_mask], self.stat_disp[baseline_mask]
        if method == 'classic':
            if n not in SPC_CONSTANTS:
                raise ValueError("For classic method, n must be between 2 and 10")
            _, A3, _, _, B3, B4 = SPC_CONSTANTS[n]
            self.cl_main, self.cl_disp = np.mean(base_xbar), np.mean(base_s)
            self.ucl_main, self.lcl_main = self.cl_main + A3 * self.cl_disp, self.cl_main - A3 * self.cl_disp
            self.ucl_disp, self.lcl_disp = B4 * self.cl_disp, B3 * self.cl_disp
        elif method == 'percentiles':
            self.cl_main, self.ucl_main, self.lcl_main = np.median(base_xbar), np.percentile(base_xbar, 99.865), np.percentile(base_xbar, 0.135)
            self.cl_disp, self.ucl_disp, self.lcl_disp = np.median(base_s), np.percentile(base_s, 99.865), np.percentile(base_s, 0.135)
        else:
            raise ValueError(f"Method {method} is currently only supported for Xbar-R chart")
        return self
