import numpy as np
from spc_lib.core.base_chart import BaseControlChart


class IMRChart(BaseControlChart):
    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.main_label = "I: Individual values"
        self.disp_label = "MR: Moving range"

    def fit(self, baseline_mask=None, method='classic'):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)
        x = np.mean(self.data, axis=1)
        mr = np.abs(np.diff(x))
        self.stat_main = x
        self.stat_disp = np.concatenate(([np.nan], mr))
        base_x = x[baseline_mask]
        base_mr = mr[baseline_mask[1:] & baseline_mask[:-1]]
        if method == 'classic':
            self.cl_main = np.mean(base_x)
            mr_bar = np.mean(base_mr)
            sigma_hat = mr_bar / 1.128
            self.ucl_main, self.lcl_main = self.cl_main + 3 * sigma_hat, self.cl_main - 3 * sigma_hat
            self.cl_disp, self.ucl_disp, self.lcl_disp = mr_bar, 3.267 * mr_bar, 0
        elif method == 'percentiles':
            self.cl_main, self.ucl_main, self.lcl_main = np.median(base_x), np.percentile(base_x, 99.865), np.percentile(base_x, 0.135)
            self.cl_disp, self.ucl_disp, self.lcl_disp = np.median(base_mr), np.percentile(base_mr, 99.865), np.percentile(base_mr, 0.135)
        elif method == 'made':
            self.cl_main = np.median(base_x)
            sigma_x = 1.4826 * np.median(np.abs(base_x - self.cl_main))
            self.ucl_main, self.lcl_main = self.cl_main + 3 * sigma_x, self.cl_main - 3 * sigma_x
            self.cl_disp = np.median(base_mr)
            sigma_mr = 1.4826 * np.median(np.abs(base_mr - self.cl_disp))
            self.ucl_disp, self.lcl_disp = self.cl_disp + 3 * sigma_mr, max(0, self.cl_disp - 3 * sigma_mr)
        elif method == 'algo_a':
            mu_star = np.median(base_x)
            s_star = 1.483 * np.median(np.abs(base_x - mu_star))
            for _ in range(3):
                x_star = np.clip(base_x, mu_star - 1.5 * s_star, mu_star + 1.5 * s_star)
                mu_star, s_star = np.mean(x_star), 1.134 * np.std(x_star, ddof=1)
            self.cl_main = mu_star
            self.ucl_main, self.lcl_main = mu_star + 3 * s_star, mu_star - 3 * s_star
            self.cl_disp, self.ucl_disp, self.lcl_disp = np.median(base_mr), np.percentile(base_mr, 99.865), np.percentile(base_mr, 0.135)
        else:
            raise ValueError(f"Unknown method: {method}")
        return self
