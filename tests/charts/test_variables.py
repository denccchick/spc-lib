import pytest
import numpy as np
from spc_lib.charts.variables import XBarRChart, XBarSChart, IMRChart, SPC_CONSTANTS


class TestXBarRChart:

    def test_init(self):
        """Test XBarRChart initialization"""
        data = np.random.randn(20, 5)
        chart = XBarRChart(data)

        assert chart.main_label == "X-bar: Subgroup means"
        assert chart.disp_label == "R: Subgroup ranges"

    def test_fit_classic(self):
        """Test XBarRChart fit with classic method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 5))

        chart = XBarRChart(data)
        chart.fit(method='classic')

        # Check main chart
        assert chart.cl_main is not None
        assert chart.ucl_main > chart.cl_main
        assert chart.lcl_main < chart.cl_main

        # Check dispersion chart
        assert chart.cl_disp is not None
        assert chart.ucl_disp > chart.cl_disp
        assert chart.lcl_disp is not None

        # Check statistics
        assert chart.stat_main.shape == (30,)
        assert chart.stat_disp.shape == (30,)

    def test_fit_percentiles(self):
        """Test XBarRChart fit with percentiles method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 5))

        chart = XBarRChart(data)
        chart.fit(method='percentiles')

        # Main chart
        assert chart.cl_main == np.median(chart.stat_main)
        assert chart.ucl_main == np.percentile(chart.stat_main, 99.865)
        assert chart.lcl_main == np.percentile(chart.stat_main, 0.135)

        # Dispersion chart
        assert chart.cl_disp == np.median(chart.stat_disp)
        assert chart.ucl_disp == np.percentile(chart.stat_disp, 99.865)
        assert chart.lcl_disp == np.percentile(chart.stat_disp, 0.135)

    def test_fit_made(self):
        """Test XBarRChart fit with MADe method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 5))

        chart = XBarRChart(data)
        chart.fit(method='made')

        assert chart.cl_main is not None
        assert chart.ucl_main > chart.cl_main
        assert chart.lcl_main < chart.cl_main
        assert chart.cl_disp is not None

    def test_fit_algo_a(self):
        """Test XBarRChart fit with Algo A method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 5))

        chart = XBarRChart(data)
        chart.fit(method='algo_a')

        assert chart.cl_main is not None
        assert chart.ucl_main > chart.cl_main
        assert chart.lcl_main < chart.cl_main

    def test_fit_with_target(self):
        """Test XBarRChart fit with target"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 5))
        target = 10.5

        chart = XBarRChart(data, target=target)
        chart.fit()

        assert chart.target == target

    def test_fit_invalid_method(self):
        """Test XBarRChart fit with invalid method"""
        data = np.random.randn(20, 5)
        chart = XBarRChart(data)

        with pytest.raises(ValueError, match="Unknown method"):
            chart.fit(method='invalid_method')

    def test_fit_invalid_n(self):
        """Test XBarRChart fit with invalid subgroup size"""
        data = np.random.randn(20, 15)  # n=15 not in constants
        chart = XBarRChart(data)

        with pytest.raises(ValueError, match="n must be between 2 and 10"):
            chart.fit(method='classic')


class TestXBarSChart:

    def test_init(self):
        """Test XBarSChart initialization"""
        data = np.random.randn(20, 5)
        chart = XBarSChart(data)

        assert chart.main_label == "X-bar: Subgroup means"
        assert chart.disp_label == "S: Subgroup standard deviations"

    def test_fit_classic(self):
        """Test XBarSChart fit with classic method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 5))

        chart = XBarSChart(data)
        chart.fit(method='classic')

        # Check main chart
        assert chart.cl_main is not None
        assert chart.ucl_main > chart.cl_main
        assert chart.lcl_main < chart.cl_main

        # Check dispersion chart
        assert chart.cl_disp is not None
        assert chart.ucl_disp > chart.cl_disp
        assert chart.lcl_disp is not None

    def test_fit_percentiles(self):
        """Test XBarSChart fit with percentiles method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 5))

        chart = XBarSChart(data)
        chart.fit(method='percentiles')

        assert chart.cl_main == np.median(chart.stat_main)
        assert chart.cl_disp == np.median(chart.stat_disp)

    def test_fit_invalid_method(self):
        """Test XBarSChart fit with invalid method"""
        data = np.random.randn(20, 5)
        chart = XBarSChart(data)

        with pytest.raises(ValueError, match="currently only supported"):
            chart.fit(method='made')

    def test_fit_with_target(self):
        """Test XBarSChart fit with target"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 5))
        target = 10.5

        chart = XBarSChart(data, target=target)
        chart.fit()

        assert chart.target == target


class TestIMRChart:

    def test_init(self):
        """Test IMRChart initialization"""
        data = np.random.randn(30, 1)
        chart = IMRChart(data)

        assert chart.main_label == "I: Individual values"
        assert chart.disp_label == "MR: Moving range"

    def test_fit_classic(self):
        """Test IMRChart fit with classic method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 1))

        chart = IMRChart(data)
        chart.fit(method='classic')

        # Check main chart
        assert chart.cl_main is not None
        assert chart.ucl_main > chart.cl_main
        assert chart.lcl_main < chart.cl_main

        # Check dispersion chart (first point is NaN)
        assert chart.stat_disp.shape == (30,)
        assert np.isnan(chart.stat_disp[0])
        assert chart.cl_disp is not None
        assert chart.ucl_disp > chart.cl_disp
        assert chart.lcl_disp == 0

    def test_fit_percentiles(self):
        """Test IMRChart fit with percentiles method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 1))

        chart = IMRChart(data)
        chart.fit(method='percentiles')

        # Main chart
        assert chart.cl_main == np.median(chart.stat_main)
        assert chart.ucl_main == np.percentile(chart.stat_main, 99.865)
        assert chart.lcl_main == np.percentile(chart.stat_main, 0.135)

    def test_fit_made(self):
        """Test IMRChart fit with MADe method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 1))

        chart = IMRChart(data)
        chart.fit(method='made')

        assert chart.cl_main is not None
        assert chart.ucl_main > chart.cl_main
        assert chart.lcl_main < chart.cl_main

    def test_fit_algo_a(self):
        """Test IMRChart fit with Algo A method"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 1))

        chart = IMRChart(data)
        chart.fit(method='algo_a')

        assert chart.cl_main is not None
        assert chart.ucl_main > chart.cl_main
        assert chart.lcl_main < chart.cl_main

    def test_fit_invalid_method(self):
        """Test IMRChart fit with invalid method"""
        data = np.random.randn(20, 1)
        chart = IMRChart(data)

        with pytest.raises(ValueError, match="Unknown method"):
            chart.fit(method='invalid_method')

    def test_fit_baseline_mask(self):
        """Test IMRChart fit with baseline mask"""
        np.random.seed(42)
        data = np.random.normal(10, 2, (30, 1))

        baseline_mask = np.array([True] * 20 + [False] * 10)

        chart = IMRChart(data)
        chart.fit(baseline_mask=baseline_mask)

        # Moving range mask should handle the baseline properly
        assert chart.cl_main is not None
