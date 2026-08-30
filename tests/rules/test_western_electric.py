import pytest
import numpy as np
from spc_lib.rules.western_electric import detect_violations, _WesternElectricRules


class TestWesternElectricRules:

    def test_rule_1_beyond_3sigma(self):
        """Test Rule 1: Point beyond 3 sigma"""
        data = np.array([0, 0, 0, 0, 4, 0, 0])  # 4 is > 3 sigma (center=0, sigma=1)
        we = _WesternElectricRules(data, center=0, sigma=1)

        violations = we._rule_1()
        assert violations == [4]

    def test_rule_2_nine_on_one_side(self):
        """Test Rule 2: 9 points on one side"""
        data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        we = _WesternElectricRules(data, center=0, sigma=1)

        violations = we._rule_2()
        assert len(violations) >= 9
        assert 0 in violations
        assert 8 in violations

    def test_rule_3_six_trend(self):
        """Test Rule 3: 6 points with trend"""
        data = np.array([0, 1, 2, 3, 4, 5, 6, 0])
        we = _WesternElectricRules(data, center=0, sigma=1)

        violations = we._rule_3()
        assert len(violations) >= 6

    def test_rule_4_fourteen_alternating(self):
        """Test Rule 4: 14 points alternating"""
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        we = _WesternElectricRules(data, center=0, sigma=1)

        violations = we._rule_4()
        assert len(violations) >= 14

    def test_rule_5_two_of_three_in_zone_a(self):
        """Test Rule 5: 2 of 3 in zone A"""
        data = np.array([2.5, 0, 2.5, 0, 0])  # 2.5 is in zone A (> 2 sigma)
        we = _WesternElectricRules(data, center=0, sigma=1)

        violations = we._rule_5()
        assert len(violations) > 0

    def test_rule_6_four_of_five_in_zone_b(self):
        """Test Rule 6: 4 of 5 in zone B"""
        data = np.array([1.5, 1.5, 1.5, 1.5, 0, 0])  # 1.5 is in zone B (> 1 sigma)
        we = _WesternElectricRules(data, center=0, sigma=1)

        violations = we._rule_6()
        assert len(violations) > 0

    def test_rule_7_fifteen_in_zone_c(self):
        """Test Rule 7: 15 points in zone C"""
        data = np.array([0.5] * 15 + [0])  # 0.5 is within zone C (< 1 sigma)
        we = _WesternElectricRules(data, center=0, sigma=1)

        violations = we._rule_7()
        assert len(violations) >= 15

    def test_rule_8_eight_outside_zone_c(self):
        """Test Rule 8: 8 points outside zone C"""
        data = np.array([1.5] * 8 + [0])  # 1.5 is outside zone C (> 1 sigma)
        we = _WesternElectricRules(data, center=0, sigma=1)

        violations = we._rule_8()
        assert len(violations) >= 8

    def test_check_all_rules(self):
        """Test checking all rules"""
        data = np.array([0, 4, 0, 0, 0, 0, 0, 0, 0, 0])  # Rule 1 violation at index 1
        we = _WesternElectricRules(data, center=0, sigma=1)

        results = we.check()
        assert 1 in results
        assert results[1] == [1]

    def test_check_specific_rules(self):
        """Test checking specific rules"""
        data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 4])
        we = _WesternElectricRules(data, center=0, sigma=1)

        results = we.check(rules=[1, 2])
        assert 1 in results
        assert 2 in results
        assert 3 not in results


class TestDetectViolations:

    def test_detect_violations_basic(self):
        """Test detect_violations function basic usage"""
        data = np.array([0, 4, 0, 0, 0, 0, 0, 0, 0, 0])

        results = detect_violations(data, center=0, sigma=1, last_n=None)

        assert isinstance(results, dict)
        assert 1 in results
        assert results[1] == [1]

    def test_detect_violations_last_n(self):
        """Test detect_violations with last_n parameter"""
        data = np.array([0, 4, 0, 0, 0, 0, 0, 0, 0, 4])

        results = detect_violations(data, center=0, sigma=1, last_n=5)

        # Only the last 5 points should be checked
        # The violation at index 1 should be excluded
        assert 1 in results
        assert 1 not in results[1]  # Index 1 is not in last 5

    def test_detect_violations_with_dates(self):
        """Test detect_violations with date filtering"""
        data = np.array([0, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        dates = np.array([
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'
        ], dtype='datetime64')  # <-- Исправлено: указываем dtype

        results = detect_violations(
            data, center=0, sigma=1,
            date_from='2024-01-05', dates=dates, last_n=None
        )

        # Violation at index 1 should be excluded (before Jan 5)
        assert 1 in results
        assert 1 not in results[1]

    def test_detect_violations_no_violations(self):
        """Test detect_violations with no violations"""
        data = np.random.normal(0, 0.5, 30)

        results = detect_violations(data, center=0, sigma=1, last_n=None)

        # All rules should be present, but may be empty
        for rule in range(1, 9):
            assert rule in results
            assert isinstance(results[rule], list)

    def test_detect_violations_requires_dates(self):
        """Test detect_violations raises error when dates missing"""
        data = np.array([0, 4, 0])

        with pytest.raises(ValueError, match="dates required"):
            detect_violations(data, center=0, sigma=1, date_from='2024-01-01', last_n=None)

    def test_detect_violations_zero_sigma(self):
        """Test detect_violations with zero sigma (handles gracefully)"""
        data = np.array([1, 1, 1, 1])

        results = detect_violations(data, center=0, sigma=0, last_n=None)

        # Should not raise error, sigma is set to 1e-10 internally
        assert isinstance(results, dict)
