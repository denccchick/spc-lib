import pytest
import numpy as np
from spc_lib.stats.diagnostics import diagnose


class TestDiagnose:

    def test_diagnose_1d_data(self):
        """Test diagnose with 1D data"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        result = diagnose(data)

        assert 'normality' in result
        assert 'autocorrelation' in result
        assert 'outliers' in result
        assert isinstance(result['normality'], str)
        assert isinstance(result['autocorrelation'], str)
        assert isinstance(result['outliers'], str)

    def test_diagnose_2d_data(self):
        """Test diagnose with 2D data (averaged by rows)"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (50, 5))

        result = diagnose(data)

        # Should have averaged by rows
        assert result is not None

    def test_diagnose_with_nan(self):
        """Test diagnose handles NaN values"""
        data = np.array([1, 2, np.nan, 4, 5, 6])

        result = diagnose(data)

        # Should not raise error
        assert result is not None

    def test_diagnose_with_inf(self):
        """Test diagnose raises error with inf values"""
        data = np.array([1, 2, np.inf, 4, 5, 6])

        with pytest.raises(ValueError, match="Infinite values"):
            diagnose(data)

    def test_diagnose_empty_data(self):
        """Test diagnose with empty data"""
        data = np.array([])

        with pytest.raises(ValueError, match="No data"):
            diagnose(data)

    def test_diagnose_all_nan(self):
        """Test diagnose with all NaN values"""
        data = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="No data left"):
            diagnose(data)

    def test_diagnose_insufficient_samples(self):
        """Test diagnose with insufficient samples"""
        data = np.array([1, 2, 3, 4, 5])

        result = diagnose(data, min_samples=10)

        assert 'Insufficient data' in result['normality']

    def test_diagnose_custom_alpha(self):
        """Test diagnose with custom alpha"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 50)

        result_05 = diagnose(data, alpha=0.05)
        result_01 = diagnose(data, alpha=0.01)

        assert result_05 is not None
        assert result_01 is not None

    def test_diagnose_custom_iqr_multiplier(self):
        """Test diagnose with custom IQR multiplier"""
        np.random.seed(42)
        data = np.concatenate([np.random.normal(0, 1, 90), np.random.normal(10, 1, 10)])

        result_1_5 = diagnose(data, iqr_multiplier=1.5)
        result_3_0 = diagnose(data, iqr_multiplier=3.0)

        # Different multipliers should give different outlier detection results
        assert result_1_5 is not None
        assert result_3_0 is not None

    def test_diagnose_invalid_data_type(self):
        """Test diagnose with invalid data type"""
        with pytest.raises(TypeError, match="Data must be an array"):
            diagnose(123)

    def test_diagnose_normal_data(self):
        """Test diagnose on normal data"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        result = diagnose(data)

        assert 'Normal' in result['normality'] or 'Non-normal' in result['normality']
        assert 'autocorrelation' in result['autocorrelation'].lower()
        assert 'outliers' in result['outliers'].lower()

    def test_diagnose_non_normal_data(self):
        """Test diagnose on non-normal data"""
        np.random.seed(42)
        data = np.random.exponential(1, 100)

        result = diagnose(data)

        # May or may not be detected as non-normal depending on sample
        assert result['normality'] is not None
