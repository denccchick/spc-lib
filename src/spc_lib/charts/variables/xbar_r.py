import numpy as np
from spc_lib.core.base_chart import BaseControlChart
from .constants import SPC_CONSTANTS


class XBarRChart(BaseControlChart):
    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.main_label = "X-bar: Subgroup means"
        self.disp_label = "R: Subgroup ranges"

    def fit(self, baseline_mask=None, method='classic'):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)
        n = self.subgroup_size
        self.stat_main = np.mean(self.data, axis=1)
        self.stat_disp = np.ptp(self.data, axis=1)
        base_xbar, base_r = self.stat_main[baseline_mask], self.stat_disp[baseline_mask]
        if method == 'classic':
            if n not in SPC_CONSTANTS:
                raise ValueError("For classic method, n must be between 2 and 10")
            A2, _, D3, D4, _, _ = SPC_CONSTANTS[n]
            self.cl_main, self.cl_disp = np.mean(base_xbar), np.mean(base_r)
            self.ucl_main, self.lcl_main = self.cl_main + A2 * self.cl_disp, self.cl_main - A2 * self.cl_disp
            self.ucl_disp, self.lcl_disp = D4 * self.cl_disp, D3 * self.cl_disp
        elif method == 'percentiles':
            self.cl_main, self.ucl_main, self.lcl_main = np.median(base_xbar), np.percentile(base_xbar, 99.865), np.percentile(base_xbar, 0.135)
            self.cl_disp, self.ucl_disp, self.lcl_disp = np.median(base_r), np.percentile(base_r, 99.865), np.percentile(base_r, 0.135)
        elif method == 'made':
            self.cl_main = np.median(base_xbar)
            self.ucl_main, self.lcl_main = self.cl_main + 3 * 1.4826 * np.median(np.abs(base_xbar - self.cl_main)), self.cl_main - 3 * 1.4826 * np.median(np.abs(base_xbar - self.cl_main))
            self.cl_disp = np.median(base_r)
            self.ucl_disp, self.lcl_disp = self.cl_disp + 3 * 1.4826 * np.median(np.abs(base_r - self.cl_disp)), self.cl_disp - 3 * 1.4826 * np.median(np.abs(base_r - self.cl_disp))
        elif method == 'algo_a':
            mu_star = np.median(base_xbar)
            s_star = 1.483 * np.median(np.abs(base_xbar - mu_star))
            for _ in range(3):
                x_star = np.clip(base_xbar, mu_star - 1.5 * s_star, mu_star + 1.5 * s_star)
                mu_star, s_star = np.mean(x_star), 1.134 * np.std(x_star, ddof=1)
            self.cl_main = mu_star
            self.ucl_main, self.lcl_main = mu_star + 3 * s_star / np.sqrt(n), mu_star - 3 * s_star / np.sqrt(n)
            self.cl_disp, self.ucl_disp, self.lcl_disp = np.median(base_r), np.percentile(base_r, 99.865), np.percentile(base_r, 0.135)
        else:
            raise ValueError(f"Unknown method: {method}")
        return self
