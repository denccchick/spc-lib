import pytest
import numpy as np
from spc_lib.charts.time_weighted import CUSUMChart, CUSUMVarianceChart, EWMAChart


class TestCUSUMChart:

    def test_init(self):
        """Test CUSUMChart initialization"""
        data = np.random.randn(30, 1)
        chart = CUSUMChart(data, h=5.0, k=0.5)

        assert chart.h == 5.0
        assert chart.k == 0.5
        assert chart.main_label == "CUSUM: Cumulative sum (upper/lower)"

    def test_fit_classic(self):
        """Test CUSUMChart fit with classic method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 1))

        chart = CUSUMChart(data, h=5.0, k=0.5)
        chart.fit(method='classic')

        assert chart.cusum_upper is not None
        assert chart.cusum_lower is not None
        assert chart.cl_main == 0
        assert chart.ucl_main == chart.h
        assert chart.lcl_main == chart.h
        assert chart.sigma_est is not None

    def test_fit_with_target(self):
        """Test CUSUMChart fit with target"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (50, 1))
        target = 10

        chart = CUSUMChart(data, target=target)
        chart.fit()

        assert chart.target == target

    def test_fit_percentiles(self):
        """Test CUSUMChart fit with percentiles method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 1))

        chart = CUSUMChart(data)
        chart.fit(method='percentiles')

        assert chart.cl_main == 0
        assert chart.ucl_main is not None
        assert chart.lcl_main is not None

    def test_fit_made(self):
        """Test CUSUMChart fit with MADe method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 1))

        chart = CUSUMChart(data)
        chart.fit(method='made')

        assert chart.cl_main == 0
        assert chart.ucl_main == chart.h
        assert chart.lcl_main == chart.h

    def test_fit_invalid_method(self):
        """Test CUSUMChart fit with invalid method"""
        data = np.random.randn(20, 1)
        chart = CUSUMChart(data)

        with pytest.raises(ValueError, match="Unknown method"):
            chart.fit(method='invalid_method')

    def test_get_cusum_stats(self):
        """Test get_cusum_stats method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (30, 1))

        chart = CUSUMChart(data)
        chart.fit()

        upper, lower = chart.get_cusum_stats()
        assert upper is chart.cusum_upper
        assert lower is chart.cusum_lower
        assert len(upper) == 30
        assert len(lower) == 30

    def test_fit_with_std_est(self):
        """Test CUSUMChart fit with provided std_est"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (30, 1))

        chart = CUSUMChart(data, std_est=1.0)
        chart.fit()

        assert chart.sigma_est == 1.0


class TestEWMAChart:

    def test_init(self):
        """Test EWMAChart initialization"""
        data = np.random.randn(30, 1)
        chart = EWMAChart(data, lambda_=0.2, L=3)

        assert chart.lambda_ == 0.2
        assert chart.L == 3
        assert chart.main_label == "EWMA: Exponentially Weighted Moving Average"

    def test_fit_classic(self):
        """Test EWMAChart fit with classic method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 1))

        chart = EWMAChart(data, lambda_=0.2, L=3)
        chart.fit(method='classic')

        assert chart.ewma_values is not None
        assert chart.ewma_sigma is not None
        assert chart.cl_main is not None
        assert chart.ucl_main is not None
        assert chart.lcl_main is not None
        assert chart.sigma_est is not None

        # First EWMA value should equal target
        assert chart.ewma_values[0] == chart.target

    def test_fit_with_target(self):
        """Test EWMAChart fit with target"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (50, 1))
        target = 10

        chart = EWMAChart(data, target=target)
        chart.fit()

        assert chart.target == target
        assert chart.ewma_values[0] == target

    def test_fit_percentiles(self):
        """Test EWMAChart fit with percentiles method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 1))

        chart = EWMAChart(data)
        chart.fit(method='percentiles')

        assert chart.cl_main == np.median(chart.ewma_values)
        assert chart.ucl_main is not None
        assert chart.lcl_main is not None

    def test_fit_made(self):
        """Test EWMAChart fit with MADe method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 1))

        chart = EWMAChart(data)
        chart.fit(method='made')

        assert chart.cl_main is not None
        assert chart.ucl_main is not None
        assert chart.lcl_main is not None

    def test_fit_invalid_method(self):
        """Test EWMAChart fit with invalid method"""
        data = np.random.randn(20, 1)
        chart = EWMAChart(data)

        with pytest.raises(ValueError, match="Unknown method"):
            chart.fit(method='invalid_method')

    def test_get_ewma_values(self):
        """Test get_ewma_values method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (30, 1))

        chart = EWMAChart(data)
        chart.fit()

        values, sigma = chart.get_ewma_values()
        assert values is chart.ewma_values
        assert sigma is chart.ewma_sigma
        assert len(values) == 30
        assert len(sigma) == 30


class TestCUSUMVarianceChart:

    def test_init(self):
        """Test CUSUMVarianceChart initialization"""
        data = np.random.randn(30, 1)
        chart = CUSUMVarianceChart(data, h=5.0, k=0.5)

        assert chart.h == 5.0
        assert chart.k == 0.5
        assert chart.main_label == "CUSUM for variance (upper/lower)"

    def test_fit_classic(self):
        """Test CUSUMVarianceChart fit with classic method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 1))

        chart = CUSUMVarianceChart(data)
        chart.fit(method='classic')

        assert chart.v_values is not None
        assert chart.cusum_upper is not None
        assert chart.cusum_lower is not None
        assert chart.cl_main == 0
        assert chart.ucl_main == chart.h
        assert chart.lcl_main == chart.h
        assert chart.sigma_est is not None

    def test_fit_with_targets(self):
        """Test CUSUMVarianceChart fit with target mean and std"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (50, 1))

        chart = CUSUMVarianceChart(data, target_mean=10, target_std=2)
        chart.fit()

        assert chart.target_mean == 10
        assert chart.target_std == 2

    def test_fit_percentiles(self):
        """Test CUSUMVarianceChart fit with percentiles method"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 1))

        chart = CUSUMVarianceChart(data)
        chart.fit(method='percentiles')

        assert chart.cl_main == 0
        assert chart.ucl_main is not None
        assert chart.lcl_main is not None

    def test_fit_invalid_method(self):
        """Test CUSUMVarianceChart fit with invalid method"""
        data = np.random.randn(20, 1)
        chart = CUSUMVarianceChart(data)

        with pytest.raises(ValueError, match="Unknown method"):
            chart.fit(method='invalid_method')
