import pytest
import numpy as np
from spc_lib.core.base_chart import BaseControlChart


class ConcreteChart(BaseControlChart):
    """Concrete implementation for testing BaseControlChart"""

    def fit(self, baseline_mask=None, method='classic'):
        self.stat_main = np.mean(self.data, axis=1)
        self.cl_main = np.mean(self.stat_main)
        self.ucl_main = self.cl_main + 3 * np.std(self.stat_main)
        self.lcl_main = self.cl_main - 3 * np.std(self.stat_main)
        self.sigma_est = np.std(self.stat_main)
        return self


class TestBaseControlChart:

    def test_init_with_2d_data(self):
        """Test initialization with 2D data"""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        chart = ConcreteChart(data)

        assert chart.n_subgroups == 3
        assert chart.subgroup_size == 3
        assert chart.data.shape == (3, 3)

    def test_init_with_1d_data(self):
        """Test initialization with 1D data"""
        data = np.array([1, 2, 3, 4, 5])
        chart = ConcreteChart(data)

        assert chart.n_subgroups == 5
        assert chart.subgroup_size == 1
        assert chart.data.shape == (5,)

    def test_init_with_datetimes(self):
        """Test initialization with datetime stamps"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        datetimes = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64')
        chart = ConcreteChart(data, datetimes=datetimes)

        assert chart.datetimes is not None
        assert len(chart.datetimes) == 3

    def test_get_mask_without_datetimes(self):
        """Test mask generation without datetimes"""
        data = np.random.randn(50, 1)
        chart = ConcreteChart(data)
        chart.stat_main = np.arange(50)

        # Test last_n
        mask = chart._get_mask(last_n=30)
        assert sum(mask) == 30
        assert mask[-30:].all()

        # Test all data
        mask = chart._get_mask(last_n=None)
        assert sum(mask) == 50

    def test_get_mask_with_datetimes(self):
        """Test mask generation with datetimes"""
        data = np.random.randn(10, 1)
        datetimes = np.array([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'
        ], dtype='datetime64')  # <-- Исправлено: указываем dtype

        chart = ConcreteChart(data, datetimes=datetimes)
        chart.stat_main = np.arange(10)

        # Test start filter
        mask = chart._get_mask(start='2024-01-05')
        assert sum(mask) == 6  # From Jan 5 to Jan 10

        # Test end filter
        mask = chart._get_mask(end='2024-01-05')
        assert sum(mask) == 5  # From Jan 1 to Jan 5

    def test_fit_raises_not_implemented(self):
        """Test that fit raises NotImplementedError on base class"""
        data = np.array([[1, 2], [3, 4]])
        chart = BaseControlChart(data)

        with pytest.raises(NotImplementedError):
            chart.fit()

    def test_capability_requires_fit(self):
        """Test that capability raises error when not fitted"""
        data = np.array([[1, 2], [3, 4]])
        chart = ConcreteChart(data)

        with pytest.raises(ValueError, match="Chart has not been fitted"):
            chart.capability(usl=10, lsl=0)

    def test_capability_requires_limits(self):
        """Test that capability raises error without specification limits"""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        chart = ConcreteChart(data)
        chart.fit()

        with pytest.raises(ValueError, match="At least one specification limit"):
            chart.capability()

    def test_capability_calculation(self):
        """Test capability indices calculation"""
        np.random.seed(42)
        data = np.random.normal(10, 1, (30, 5))
        chart = ConcreteChart(data)
        chart.fit()

        # Test with both USL and LSL
        result = chart.capability(usl=13, lsl=7)

        assert 'cp' in result
        assert 'cpk' in result
        assert 'cpl' in result
        assert 'cpu' in result
        assert not np.isnan(result['cp'])

    def test_capability_with_single_limit(self):
        """Test capability with only USL or only LSL"""
        np.random.seed(42)
        data = np.random.normal(10, 1, (30, 5))
        chart = ConcreteChart(data)
        chart.fit()

        # Only USL
        result = chart.capability(usl=13)
        assert np.isnan(result['cp'])
        assert not np.isnan(result['cpu'])
        assert np.isnan(result['cpl'])
        assert not np.isnan(result['cpk'])

        # Only LSL
        result = chart.capability(lsl=7)
        assert np.isnan(result['cp'])
        assert np.isnan(result['cpu'])
        assert not np.isnan(result['cpl'])
        assert not np.isnan(result['cpk'])
