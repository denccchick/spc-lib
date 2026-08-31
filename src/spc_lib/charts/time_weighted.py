import numpy as np
import warnings
from spc_lib.core.base_chart import BaseControlChart


class CUSUMChart(BaseControlChart):
    """
    Cumulative Sum (CUSUM) control chart for detecting small process shifts.

    Parameters
    ----------
    data : array-like
        Subgroup data (2D array) or individual values (1D)
    datetimes : array-like, optional
        Time stamps
    target : float, optional
        Target process value
    usl, lsl : float, optional
        Upper/lower specification limits
    h : float, default=5
        Decision interval parameter (distance from zero line to boundary)
    k : float, default=0.5
        Reference value parameter (typically 0.5 for standard CUSUM)
    std_est : float, optional
        Estimated standard deviation. If None, estimated from data.
    """

    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None,
                 h=5.0, k=0.5, std_est=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.h = h
        self.k = k
        self.std_est = std_est
        self.main_label = "CUSUM: Cumulative sum (upper/lower)"
        self.disp_label = None

        self.cusum_upper = None
        self.cusum_lower = None
        self.sigma_est = None

    def fit(self, baseline_mask=None, method='classic'):
        """
        Fit the CUSUM chart to the data.

        Methods:
        - 'classic': standard CUSUM with h and k parameters
        - 'percentiles': empirical limits based on percentiles
        - 'made': robust CUSUM based on MAD
        """
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        if self.data.ndim == 2:
            if self.subgroup_size == 1:
                x = self.data.flatten()
                n = 1
            else:
                x = np.mean(self.data, axis=1)
                n = self.subgroup_size
        else:
            warnings.warn(
                "1D array provided. It is recommended to use 2D format: data.reshape(-1, 1)",
                UserWarning
            )
            x = self.data
            n = 1

        base_x = x[baseline_mask]

        if self.std_est is None:
            if n > 1 and method == 'classic':
                r = np.ptp(self.data[baseline_mask], axis=1)
                d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
                      6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
                sigma_est = np.mean(r) / d2.get(n, 1.128)
            else:
                sigma_est = np.std(base_x, ddof=1)
        else:
            sigma_est = self.std_est

        if self.target is not None:
            target = self.target
        elif method == 'classic':
            target = np.mean(base_x)
        else:
            target = np.median(base_x)

        # Normalization
        if sigma_est > 0:
            z = (x - target) / sigma_est
        else:
            z = x - target
            warnings.warn("Sigma is zero, using unnormalized statistic", UserWarning)

        self.cusum_upper = np.zeros(len(x))
        self.cusum_lower = np.zeros(len(x))

        if method == 'classic':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i-1] + z[i] - self.k)
                self.cusum_lower[i] = max(0, self.cusum_lower[i-1] - z[i] - self.k)

            self.cl_main = 0
            self.ucl_main = self.h
            self.lcl_main = self.h

        elif method == 'percentiles':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i-1] + z[i])
                self.cusum_lower[i] = max(0, self.cusum_lower[i-1] - z[i])

            base_cusum_upper = self.cusum_upper[baseline_mask]
            base_cusum_lower = self.cusum_lower[baseline_mask]

            self.cl_main = 0
            self.ucl_main = np.percentile(base_cusum_upper, 99.865)
            self.lcl_main = np.percentile(base_cusum_lower, 99.865)

        elif method == 'made':
            mad = np.median(np.abs(base_x - np.median(base_x)))
            robust_sigma = 1.4826 * mad if mad > 0 else 1e-10

            z_robust = (x - target) / robust_sigma if robust_sigma > 0 else x - target

            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i-1] + z_robust[i] - self.k)
                self.cusum_lower[i] = max(0, self.cusum_lower[i-1] - z_robust[i] - self.k)

            self.cl_main = 0
            self.ucl_main = self.h
            self.lcl_main = self.h
            sigma_est = robust_sigma

        else:
            raise ValueError(f"Unknown method: {method}")

        self.target = target
        self.sigma_est = sigma_est
        self.stat_main = x

        return self

    def get_cusum_stats(self):
        """Return the upper and lower CUSUM statistics"""
        return self.cusum_upper, self.cusum_lower


class EWMAChart(BaseControlChart):
    """
    Exponentially Weighted Moving Average (EWMA) control chart
    for detecting small process shifts.

    Parameters
    ----------
    data : array-like
        Subgroup data (2D array) or individual values (1D)
    datetimes : array-like, optional
        Time stamps
    target : float, optional
        Target process value
    usl, lsl : float, optional
        Upper/lower specification limits
    lambda_ : float, default=0.2
        Weighting factor (0 < lambda <= 1)
    L : float, default=3
        Control limit width coefficient
    std_est : float, optional
        Estimated standard deviation. If None, estimated from data.
    """

    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None,
                 lambda_=0.2, L=3, std_est=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.lambda_ = lambda_
        self.L = L
        self.std_est = std_est
        self.main_label = "EWMA: Exponentially Weighted Moving Average"
        self.disp_label = None

        self.ewma_values = None
        self.ewma_sigma = None
        self.sigma_est = None

    def fit(self, baseline_mask=None, method='classic'):
        """
        Fit the EWMA chart to the data.

        Methods:
        - 'classic': standard EWMA with fixed limits
        - 'percentiles': empirical limits based on percentiles
        - 'made': robust EWMA based on MAD
        """
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        # Extract data
        if self.data.ndim == 2:
            if self.subgroup_size == 1:
                x = self.data.flatten()
                n = 1
            else:
                x = np.mean(self.data, axis=1)
                n = self.subgroup_size
        else:
            warnings.warn(
                "1D array provided. It is recommended to use 2D format: data.reshape(-1, 1)",
                UserWarning
            )
            x = self.data
            n = 1

        base_x = x[baseline_mask]

        # Estimate standard deviation
        if self.std_est is None:
            if n > 1 and method == 'classic':
                r = np.ptp(self.data[baseline_mask], axis=1)
                d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
                      6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
                sigma_est = np.mean(r) / d2.get(n, 1.128)
            else:
                sigma_est = np.std(base_x, ddof=1)
        else:
            sigma_est = self.std_est

        # Target value
        if self.target is not None:
            target = self.target
        elif method == 'classic':
            target = np.mean(base_x)
        else:
            target = np.median(base_x)

        lambda_ = self.lambda_
        L = self.L

        # Calculate EWMA
        self.ewma_values = np.zeros(len(x))
        self.ewma_values[0] = target

        for i in range(1, len(x)):
            self.ewma_values[i] = lambda_ * x[i] + (1 - lambda_) * self.ewma_values[i-1]

        # EWMA standard deviation: σ * sqrt(λ/(2-λ) * (1 - (1-λ)^(2i)))
        weights = np.zeros(len(x))
        for i in range(1, len(x) + 1):
            weights[i-1] = np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_)**(2*i)))

        self.ewma_sigma = sigma_est * weights

        # Calculate limits based on method
        if method == 'classic':
            self.cl_main = target
            self.ucl_main = target + L * self.ewma_sigma
            self.lcl_main = target - L * self.ewma_sigma

        elif method == 'percentiles':
            base_ewma = self.ewma_values[baseline_mask]
            self.cl_main = np.median(base_ewma)
            self.ucl_main = np.percentile(base_ewma, 99.865)
            self.lcl_main = np.percentile(base_ewma, 0.135)

        elif method == 'made':
            mad = np.median(np.abs(base_x - np.median(base_x)))
            robust_sigma = 1.4826 * mad if mad > 0 else 1e-10

            self.ewma_sigma = robust_sigma * weights
            self.cl_main = target
            self.ucl_main = target + L * self.ewma_sigma
            self.lcl_main = target - L * self.ewma_sigma
            sigma_est = robust_sigma

        else:
            raise ValueError(f"Unknown method: {method}")

        self.target = target
        self.sigma_est = sigma_est
        self.stat_main = self.ewma_values

        return self

    def get_ewma_values(self):
        """Return the EWMA values and their standard deviations"""
        return self.ewma_values, self.ewma_sigma


class CUSUMVarianceChart(BaseControlChart):
    """
    CUSUM chart for monitoring variance/standard deviation.
    Uses Hawkins' transformation for monitoring process variability.

    Parameters
    ----------
    data : array-like
        Subgroup data (2D array) or individual values (1D)
    datetimes : array-like, optional
        Time stamps
    target_mean : float, optional
        Target process mean. If None - estimated from data.
    target_std : float, optional
        Target standard deviation. If None - estimated from data.
    usl, lsl : float, optional
        Upper/lower specification limits
    h : float, default=5
        Decision interval parameter
    k : float, default=0.5
        Reference value parameter
    """

    def __init__(self, data, datetimes=None, target_mean=None, target_std=None,
                 usl=None, lsl=None, h=5.0, k=0.5):
        super().__init__(data, datetimes, None, usl, lsl)  # target passed as None, using target_mean
        self.target_mean = target_mean
        self.target_std = target_std
        self.h = h
        self.k = k
        self.main_label = "CUSUM for variance (upper/lower)"
        self.disp_label = None

        self.cusum_upper = None
        self.cusum_lower = None
        self.v_values = None  # Transformed values v_i
        self.sigma_est = None

    def fit(self, baseline_mask=None, method='classic'):
        """
        Fit the variance CUSUM chart to the data.

        Methods:
        - 'classic': standard CUSUM for variance with Hawkins transformation
        - 'percentiles': empirical limits based on percentiles
        """
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        # Extract data
        if self.data.ndim == 2:
            if self.subgroup_size == 1:
                x = self.data.flatten()
                n = 1
            else:
                x = np.mean(self.data, axis=1)
                n = self.subgroup_size
        else:
            warnings.warn(
                "1D array provided. It is recommended to use 2D format: data.reshape(-1, 1)",
                UserWarning
            )
            x = self.data
            n = 1

        base_x = x[baseline_mask]

        # Estimate mean and standard deviation
        if self.target_mean is not None:
            mu = self.target_mean
        elif method == 'classic':
            mu = np.mean(base_x)
        else:
            mu = np.median(base_x)

        if self.target_std is not None:
            sigma = self.target_std
        elif method == 'classic' and n > 1:
            r = np.ptp(self.data[baseline_mask], axis=1)
            d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
                  6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
            sigma = np.mean(r) / d2.get(n, 1.128)
        else:
            sigma = np.std(base_x, ddof=1)

        # Standardized values
        if sigma > 0:
            y = (x - mu) / sigma
        else:
            y = x - mu
            warnings.warn("Sigma is zero, using unnormalized statistic", UserWarning)

        # Hawkins transformation for monitoring variance
        # v_i = (sqrt(|y_i|) - 0.822) / 0.349
        # Under normal distribution v_i ~ N(0,1)
        self.v_values = (np.sqrt(np.abs(y)) - 0.822) / 0.349

        self.cusum_upper = np.zeros(len(x))
        self.cusum_lower = np.zeros(len(x))

        if method == 'classic':
            # Standard CUSUM for v_i
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i-1] + self.v_values[i] - self.k)
                self.cusum_lower[i] = max(0, self.cusum_lower[i-1] - self.k - self.v_values[i])

            self.cl_main = 0
            self.ucl_main = self.h
            self.lcl_main = self.h

        elif method == 'percentiles':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i-1] + self.v_values[i])
                self.cusum_lower[i] = max(0, self.cusum_lower[i-1] - self.v_values[i])

            base_cusum_upper = self.cusum_upper[baseline_mask]
            base_cusum_lower = self.cusum_lower[baseline_mask]

            self.cl_main = 0
            self.ucl_main = np.percentile(base_cusum_upper, 99.865)
            self.lcl_main = np.percentile(base_cusum_lower, 99.865)

        else:
            raise ValueError(f"Unknown method: {method}")

        self.target = mu  # For compatibility with base class
        self.target_mean = mu
        self.target_std = sigma
        self.sigma_est = sigma
        self.stat_main = x  # Original data for reference

        return self
