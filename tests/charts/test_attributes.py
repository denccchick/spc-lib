import pytest
import numpy as np
from spc_lib.charts.attributes import PChart, CChart


class TestPChart:

    def test_init_with_fixed_n(self):
        """Test PChart initialization with fixed sample size"""
        data = np.array([0.1, 0.2, 0.15, 0.25, 0.1])
        chart = PChart(data, n_fixed=100)

        assert chart.n_fixed == 100
        assert chart.main_label == "p-chart: Proportion of nonconforming units"

    def test_init_with_variable_n(self):
        """Test PChart initialization with variable sample size"""
        data = np.array([[5, 100], [8, 120], [6, 110], [10, 130]])
        chart = PChart(data)

        assert chart.n_fixed is None
        assert chart.data.shape == (4, 2)

    def test_fit_fixed_n_classic(self):
        """Test PChart fit with fixed n and classic method"""
        np.random.seed(42)
        p_true = 0.1
        n = 100
        data = np.random.binomial(n, p_true, 30) / n

        chart = PChart(data, n_fixed=n)
        chart.fit(method='classic')

        assert chart.p_bar is not None
        assert chart.cl_main == chart.p_bar
        assert chart.ucl_main > chart.cl_main
        assert chart.lcl_main < chart.cl_main
        assert chart.sigma_est is not None

    def test_fit_variable_n_classic(self):
        """Test PChart fit with variable n and classic method"""
        data = np.array([
            [5, 100], [8, 120], [6, 110], [10, 130],
            [7, 105], [9, 115], [4, 95], [11, 125]
        ])

        chart = PChart(data)
        chart.fit(method='classic')

        assert chart.p_values is not None
        assert chart.n_values is not None
        assert chart.p_bar is not None
        assert chart.n_bar is not None
        assert len(chart.ucl_main) == len(chart.p_values)
        assert len(chart.lcl_main) == len(chart.p_values)

    def test_fit_percentiles_method(self):
        """Test PChart fit with percentiles method"""
        np.random.seed(42)
        data = np.random.binomial(100, 0.1, 50) / 100

        chart = PChart(data, n_fixed=100)
        chart.fit(method='percentiles')

        assert chart.cl_main == np.median(data)
        assert chart.ucl_main == np.percentile(data, 99.865)
        assert chart.lcl_main == np.percentile(data, 0.135)

    def test_fit_use_average_n(self):
        """Test PChart fit with average n for variable limits"""
        data = np.array([
            [5, 100], [8, 120], [6, 110], [10, 130]
        ])

        chart = PChart(data)
        chart.fit(method='classic', use_average_n=True)

        # Should have constant limits
        assert isinstance(chart.ucl_main, (int, float))
        assert isinstance(chart.lcl_main, (int, float))

    def test_fit_invalid_method(self):
        """Test PChart fit with invalid method"""
        data = np.array([0.1, 0.2, 0.15])
        chart = PChart(data, n_fixed=100)

        with pytest.raises(ValueError, match="Unknown method"):
            chart.fit(method='invalid_method')

    def test_fit_invalid_data_shape(self):
        """Test PChart fit with invalid data shape for variable n"""
        data = np.array([0.1, 0.2, 0.3])  # 1D without n_fixed

        chart = PChart(data)
        with pytest.raises(ValueError, match="For variable sample size"):
            chart.fit()

    def test_fit_baseline_mask(self):
        """Test PChart fit with baseline mask"""
        np.random.seed(42)
        data = np.random.binomial(100, 0.1, 30) / 100

        # Use first 20 points as baseline
        baseline_mask = np.array([True] * 20 + [False] * 10)

        chart = PChart(data, n_fixed=100)
        chart.fit(baseline_mask=baseline_mask)

        # p_bar should be based only on baseline
        baseline_data = data[baseline_mask]
        expected_p_bar = np.mean(baseline_data)
        assert np.isclose(chart.p_bar, expected_p_bar, rtol=1e-2)


class TestCChart:

    def test_init(self):
        """Test CChart initialization"""
        data = np.array([2, 3, 1, 4, 2, 5])
        chart = CChart(data)

        assert chart.main_label == "c-chart: Number of defects"
        assert chart.c_values is None
        assert chart.c_bar is None

    def test_fit_classic(self):
        """Test CChart fit with classic method"""
        np.random.seed(42)
        c_true = 3
        data = np.random.poisson(c_true, 30)

        chart = CChart(data)
        chart.fit(method='classic')

        assert chart.c_bar is not None
        assert chart.cl_main == chart.c_bar
        assert chart.ucl_main == chart.c_bar + 3 * np.sqrt(chart.c_bar)
        assert chart.lcl_main == max(0, chart.c_bar - 3 * np.sqrt(chart.c_bar))
        assert chart.sigma_est == np.sqrt(chart.c_bar)

    def test_fit_percentiles(self):
        """Test CChart fit with percentiles method"""
        data = np.array([1, 2, 3, 2, 4, 3, 2, 5, 3, 2])

        chart = CChart(data)
        chart.fit(method='percentiles')

        assert chart.cl_main == np.median(data)
        assert chart.ucl_main == np.percentile(data, 99.865)
        assert chart.lcl_main == np.percentile(data, 0.135)

    def test_fit_2d_data(self):
        """Test CChart fit with 2D data (n, 1)"""
        data = np.array([[2], [3], [1], [4], [2]])

        chart = CChart(data)
        chart.fit()

        assert chart.c_values is not None
        assert len(chart.c_values) == 5

    def test_fit_invalid_method(self):
        """Test CChart fit with invalid method"""
        data = np.array([1, 2, 3])
        chart = CChart(data)

        with pytest.raises(ValueError, match="Unknown method"):
            chart.fit(method='invalid_method')

    def test_fit_baseline_mask(self):
        """Test CChart fit with baseline mask"""
        data = np.array([2, 3, 1, 4, 2, 5, 3, 4, 2, 3])

        # Use first 5 points as baseline
        baseline_mask = np.array([True] * 5 + [False] * 5)

        chart = CChart(data)
        chart.fit(baseline_mask=baseline_mask)

        expected_c_bar = np.mean(data[baseline_mask])
        assert np.isclose(chart.c_bar, expected_c_bar)

    def test_lcl_non_negative(self):
        """Test that LCL is never negative"""
        data = np.array([0, 1, 0, 0, 1])  # Low defect count

        chart = CChart(data)
        chart.fit()

        assert chart.lcl_main >= 0
