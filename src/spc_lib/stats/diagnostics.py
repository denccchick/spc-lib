import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson


def diagnose(data, alpha=0.05, iqr_multiplier=1.5, min_samples=15):
    """
    Diagnostic analysis of data distribution before building control charts.

    Parameters
    ----------
    data : array-like
        Vector (1D) or matrix (2D). If matrix — averaged by rows.
    alpha : float, default=0.05
        Significance level for statistical tests.
    iqr_multiplier : float, default=1.5
        Multiplier for IQR (1.5 - classic outlier threshold).
    min_samples : int, default=15
        Minimum number of observations for performing tests.

    Returns
    -------
    dict
        Diagnostic results: normality, autocorrelation, outliers.
    """
    # 1. Validation
    if not hasattr(data, '__len__') and not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f"Data must be an array, got {type(data).__name__}")

    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        raise ValueError("No data for analysis (empty array)")

    # If 2D — average by rows (axis=1), since X-bar charts work with means
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    # Clean infinities
    if np.any(np.isinf(data)):
        n_inf = np.isinf(data).sum()
        raise ValueError(f"Infinite values (inf) detected: {n_inf}")

    # Clean NaNs
    n_nan = np.isnan(data).sum()
    if n_nan > 0:
        print(f"Warning: removed {n_nan} missing values (NaN)")
        data = data[~np.isnan(data)]

    n = len(data)

    if n == 0:
        raise ValueError("No data left after removing missing values (NaN)")

    # Check minimum sample size
    if n < min_samples:
        msg = f'Insufficient data (n={n} < {min_samples})'
        return {'normality': msg, 'autocorrelation': msg, 'outliers': msg}

    # Check for zero variance
    if np.all(data == data[0]):
        msg = 'No variability (all values are identical)'
        return {'normality': msg, 'autocorrelation': msg, 'outliers': msg}

    # 2. Tests
    normality = _check_normality(data, alpha)
    autocorr = _check_autocorrelation(data)
    outliers = _check_outliers(data, iqr_multiplier)

    return {
        'normality': normality,
        'autocorrelation': autocorr,
        'outliers': outliers
    }


def _check_normality(data, alpha):
    """Check normality (Shapiro-Wilk or Kolmogorov-Smirnov for large data)"""
    n = len(data)

    if n > 5000:
        # For large samples, Shapiro-Wilk may be inaccurate, use K-S test
        data_std = (data - np.mean(data)) / np.std(data, ddof=1)
        _, p_value = stats.kstest(data_std, 'norm')
    else:
        _, p_value = stats.shapiro(data)

    return 'Normal' if p_value >= alpha else 'Non-normal'


def _check_autocorrelation(data):
    """Check autocorrelation (Durbin-Watson test from statsmodels)"""
    # Durbin-Watson tests residuals. For simple series, residuals = data - mean.
    residuals = data - np.mean(data)
    dw_stat = durbin_watson(residuals)

    if 1.5 <= dw_stat <= 2.5:
        return 'No autocorrelation'
    elif dw_stat < 1.5:
        return 'Positive autocorrelation'
    else:
        return 'Negative autocorrelation'


def _check_outliers(data, multiplier):
    """Detect outliers using the interquartile range (IQR) method"""
    n = len(data)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    if iqr == 0:
        return 'IQR = 0 (cannot detect outliers with classic method)'

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    # Count points beyond the whiskers
    outliers_count = np.sum((data < lower) | (data > upper))
    percentage = (outliers_count / n) * 100

    if percentage < 1.0:
        return 'Almost no outliers (<1%)'
    elif percentage < 5.0:
        return 'Normal number of outliers (1-5%)'
    elif percentage < 10.0:
        return 'Many outliers (5-10%)'
    else:
        return 'Very many outliers (>10%, heavy tails)'
