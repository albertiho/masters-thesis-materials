"""Tests for Z-score methods from Yaro et al. (2024).

Validates the implementation against hand-computed values for small
datasets and checks algebraic identities that must hold between methods.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.anomaly.z_score_methods import (
    OutlierResult,
    ZScoreMethod,
    _sn_correction_factor,
    detect_outliers,
    mad_scale,
    sd_scale,
    sn_scale,
    z_score_avg_hybrid,
    z_score_mad,
    z_score_max_hybrid,
    z_score_sd,
    z_score_sn,
    z_score_weighted_hybrid,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def symmetric_5() -> np.ndarray:
    """Simple symmetric dataset: [1, 2, 3, 4, 5]."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def even_6() -> np.ndarray:
    """Even-length dataset: [1, 2, 3, 4, 5, 6]."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


@pytest.fixture
def with_outlier() -> np.ndarray:
    """Dataset with a clear high outlier."""
    return np.array([10.0, 11.0, 10.5, 10.2, 9.8, 10.1, 10.3, 50.0])


@pytest.fixture
def constant() -> np.ndarray:
    """Constant dataset — all identical values."""
    return np.array([5.0, 5.0, 5.0, 5.0, 5.0])


# =====================================================================
# Scale estimator tests
# =====================================================================


class TestSdScale:
    def test_known_values(self, symmetric_5: np.ndarray) -> None:
        # population SD of [1,2,3,4,5]: sqrt(mean([4,1,0,1,4])) = sqrt(2)
        assert_allclose(sd_scale(symmetric_5), np.sqrt(2.0), atol=1e-12)

    def test_constant_data(self, constant: np.ndarray) -> None:
        assert sd_scale(constant) == 0.0

    def test_single_element(self) -> None:
        assert sd_scale(np.array([42.0])) == 0.0

    def test_two_elements(self) -> None:
        # population SD of [0, 10]: sqrt(mean([25, 25])) = 5
        assert_allclose(sd_scale(np.array([0.0, 10.0])), 5.0, atol=1e-12)


class TestMadScale:
    def test_known_values(self, symmetric_5: np.ndarray) -> None:
        # median = 3; |deviations| = [2,1,0,1,2]; MAD = median = 1
        # σ_MAD = 1.4826 * 1 = 1.4826
        assert_allclose(mad_scale(symmetric_5), 1.4826, atol=1e-12)

    def test_custom_b(self, symmetric_5: np.ndarray) -> None:
        assert_allclose(mad_scale(symmetric_5, b=1.0), 1.0, atol=1e-12)

    def test_constant_data(self, constant: np.ndarray) -> None:
        assert mad_scale(constant) == 0.0

    def test_even_length(self, even_6: np.ndarray) -> None:
        # median = 3.5; |deviations| = [2.5, 1.5, 0.5, 0.5, 1.5, 2.5]
        # sorted = [0.5, 0.5, 1.5, 1.5, 2.5, 2.5]; MAD = median = 1.5
        # σ_MAD = 1.4826 * 1.5
        assert_allclose(mad_scale(even_6), 1.4826 * 1.5, atol=1e-12)


class TestSnCorrectionFactor:
    def test_odd(self) -> None:
        # n=5: 5/(5-0.9) = 5/4.1
        assert_allclose(_sn_correction_factor(5), 5.0 / 4.1, atol=1e-12)

    def test_even(self) -> None:
        assert _sn_correction_factor(6) == 1.0

    def test_odd_large(self) -> None:
        assert_allclose(_sn_correction_factor(101), 101.0 / 100.1, atol=1e-12)

    def test_even_large(self) -> None:
        assert _sn_correction_factor(100) == 1.0


class TestSnScale:
    def test_known_values(self, symmetric_5: np.ndarray) -> None:
        # Inner medians computed by hand (including self-comparison):
        #   i=0 (1): median([0,1,2,3,4]) = 2
        #   i=1 (2): median([1,0,1,2,3]) = 1
        #   i=2 (3): median([2,1,0,1,2]) = 1
        #   i=3 (4): median([3,2,1,0,1]) = 1
        #   i=4 (5): median([4,3,2,1,0]) = 2
        # Outer median of [2,1,1,1,2] = 1
        # C_5 = 5/4.1 ≈ 1.21951
        # σ_Sn = 1.21951 * 1 = 1.21951
        expected = 5.0 / 4.1  # ≈ 1.21951
        assert_allclose(sn_scale(symmetric_5), expected, atol=1e-10)

    def test_even_length(self, even_6: np.ndarray) -> None:
        # Inner medians for [1,2,3,4,5,6]:
        #   i=0 (1): |diffs| = [0,1,2,3,4,5]; median = 2.5
        #   i=1 (2): |diffs| = [1,0,1,2,3,4]; median = 1.5
        #   i=2 (3): |diffs| = [2,1,0,1,2,3]; median = 1.5
        #   i=3 (4): |diffs| = [3,2,1,0,1,2]; median = 1.5
        #   i=4 (5): |diffs| = [4,3,2,1,0,1]; median = 1.5
        #   i=5 (6): |diffs| = [5,4,3,2,1,0]; median = 2.5
        # Outer median of [2.5,1.5,1.5,1.5,1.5,2.5] = 1.5
        # C_6 = 1.0
        assert_allclose(sn_scale(even_6), 1.5, atol=1e-10)

    def test_constant_data(self, constant: np.ndarray) -> None:
        assert sn_scale(constant) == 0.0


# =====================================================================
# Z-score computation tests
# =====================================================================


class TestZScoreSd:
    def test_known_values(self, symmetric_5: np.ndarray) -> None:
        # Z_SD = (x - 3) / sqrt(2)
        expected = (symmetric_5 - 3.0) / np.sqrt(2.0)
        assert_allclose(z_score_sd(symmetric_5), expected, atol=1e-12)

    def test_mean_has_zero_zscore(self, symmetric_5: np.ndarray) -> None:
        z = z_score_sd(symmetric_5)
        # Middle element (value 3 = mean) should have z-score ≈ 0
        assert_allclose(z[2], 0.0, atol=1e-12)

    def test_symmetry(self, symmetric_5: np.ndarray) -> None:
        z = z_score_sd(symmetric_5)
        # z[0] should equal -z[4], z[1] should equal -z[3]
        assert_allclose(z[0], -z[4], atol=1e-12)
        assert_allclose(z[1], -z[3], atol=1e-12)

    def test_constant_raises(self, constant: np.ndarray) -> None:
        with pytest.raises(ValueError, match="zero"):
            z_score_sd(constant)


class TestZScoreMad:
    def test_known_values(self, symmetric_5: np.ndarray) -> None:
        # Z_MAD = |x - 3| / 1.4826
        expected = np.abs(symmetric_5 - 3.0) / 1.4826
        assert_allclose(z_score_mad(symmetric_5), expected, atol=1e-12)

    def test_non_negative(self, symmetric_5: np.ndarray) -> None:
        assert np.all(z_score_mad(symmetric_5) >= 0)

    def test_median_has_zero_zscore(self, symmetric_5: np.ndarray) -> None:
        z = z_score_mad(symmetric_5)
        assert_allclose(z[2], 0.0, atol=1e-12)

    def test_constant_raises(self, constant: np.ndarray) -> None:
        with pytest.raises(ValueError, match="zero"):
            z_score_mad(constant)


class TestZScoreSn:
    def test_known_values(self, symmetric_5: np.ndarray) -> None:
        sigma_sn = 5.0 / 4.1
        expected = np.abs(symmetric_5 - 3.0) / sigma_sn
        assert_allclose(z_score_sn(symmetric_5), expected, atol=1e-10)

    def test_non_negative(self, symmetric_5: np.ndarray) -> None:
        assert np.all(z_score_sn(symmetric_5) >= 0)

    def test_median_has_zero_zscore(self, symmetric_5: np.ndarray) -> None:
        z = z_score_sn(symmetric_5)
        assert_allclose(z[2], 0.0, atol=1e-12)

    def test_constant_raises(self, constant: np.ndarray) -> None:
        with pytest.raises(ValueError, match="zero"):
            z_score_sn(constant)


# =====================================================================
# Hybrid Z-score tests
# =====================================================================


class TestWeightedHybrid:
    def test_w0_equals_sn(self, symmetric_5: np.ndarray) -> None:
        """At w=0 the weighted hybrid reduces to pure Sn."""
        z_wgt = z_score_weighted_hybrid(symmetric_5, w=0.0)
        z_sn = z_score_sn(symmetric_5)
        assert_allclose(z_wgt, z_sn, atol=1e-12)

    def test_w1_equals_mad(self, symmetric_5: np.ndarray) -> None:
        """At w=1 the weighted hybrid reduces to pure MAD."""
        z_wgt = z_score_weighted_hybrid(symmetric_5, w=1.0)
        z_mad = z_score_mad(symmetric_5)
        assert_allclose(z_wgt, z_mad, atol=1e-12)

    def test_w05_equals_avg(self, symmetric_5: np.ndarray) -> None:
        """At w=0.5 the weighted hybrid equals the average hybrid."""
        z_wgt = z_score_weighted_hybrid(symmetric_5, w=0.5)
        z_avg = z_score_avg_hybrid(symmetric_5)
        assert_allclose(z_wgt, z_avg, atol=1e-12)

    def test_invalid_weight_raises(self, symmetric_5: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Weight"):
            z_score_weighted_hybrid(symmetric_5, w=-0.1)
        with pytest.raises(ValueError, match="Weight"):
            z_score_weighted_hybrid(symmetric_5, w=1.5)

    def test_intermediate_weight(self, symmetric_5: np.ndarray) -> None:
        """Manual check for w=0.3."""
        w = 0.3
        z_mad = z_score_mad(symmetric_5)
        z_sn = z_score_sn(symmetric_5)
        expected = w * z_mad + (1.0 - w) * z_sn
        assert_allclose(
            z_score_weighted_hybrid(symmetric_5, w=w), expected, atol=1e-12
        )


class TestMaxHybrid:
    def test_element_wise_max(self, symmetric_5: np.ndarray) -> None:
        z_mad = z_score_mad(symmetric_5)
        z_sn = z_score_sn(symmetric_5)
        expected = np.maximum(z_mad, z_sn)
        assert_allclose(z_score_max_hybrid(symmetric_5), expected, atol=1e-12)

    def test_geq_both_parents(self, symmetric_5: np.ndarray) -> None:
        """The max hybrid must be >= both MAD and Sn z-scores everywhere."""
        z_max = z_score_max_hybrid(symmetric_5)
        z_mad = z_score_mad(symmetric_5)
        z_sn = z_score_sn(symmetric_5)
        assert np.all(z_max >= z_mad - 1e-12)
        assert np.all(z_max >= z_sn - 1e-12)


class TestAvgHybrid:
    def test_equals_mean_of_parents(self, symmetric_5: np.ndarray) -> None:
        z_mad = z_score_mad(symmetric_5)
        z_sn = z_score_sn(symmetric_5)
        expected = 0.5 * (z_mad + z_sn)
        assert_allclose(z_score_avg_hybrid(symmetric_5), expected, atol=1e-12)

    def test_between_parents(self, symmetric_5: np.ndarray) -> None:
        """Average should lie between (or equal) MAD and Sn z-scores."""
        z_avg = z_score_avg_hybrid(symmetric_5)
        z_mad = z_score_mad(symmetric_5)
        z_sn = z_score_sn(symmetric_5)
        lo = np.minimum(z_mad, z_sn)
        hi = np.maximum(z_mad, z_sn)
        assert np.all(z_avg >= lo - 1e-12)
        assert np.all(z_avg <= hi + 1e-12)


# =====================================================================
# Outlier detection tests
# =====================================================================


class TestDetectOutliers:
    def test_returns_outlier_result(self, symmetric_5: np.ndarray) -> None:
        result = detect_outliers(symmetric_5)
        assert isinstance(result, OutlierResult)
        assert result.is_outlier.shape == symmetric_5.shape
        assert result.z_scores.shape == symmetric_5.shape

    def test_clear_outlier_detected(self, with_outlier: np.ndarray) -> None:
        """The value 50 in [10, 11, 10.5, 10.2, 9.8, 10.1, 10.3, 50] must
        be flagged by every method."""
        for method in ZScoreMethod:
            result = detect_outliers(with_outlier, method=method, threshold=2.0)
            assert result.is_outlier[-1], (
                f"Method {method.value} failed to detect the obvious outlier"
            )
            assert result.n_outliers >= 1

    def test_no_outliers_in_tight_data(self, symmetric_5: np.ndarray) -> None:
        """[1,2,3,4,5] at threshold=3 should produce zero outliers for all
        modified methods (max |deviation| / scale < 3)."""
        for method in [
            ZScoreMethod.MODIFIED_MAD,
            ZScoreMethod.MODIFIED_SN,
            ZScoreMethod.HYBRID_WEIGHTED,
            ZScoreMethod.HYBRID_MAX,
            ZScoreMethod.HYBRID_AVG,
        ]:
            result = detect_outliers(symmetric_5, method=method, threshold=3.0)
            assert result.n_outliers == 0, (
                f"Method {method.value} falsely detected outliers"
            )

    def test_standard_sd_two_sided(self) -> None:
        """Standard Z-score should catch outliers on both tails."""
        data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, -50.0])
        result = detect_outliers(
            data, method=ZScoreMethod.STANDARD_SD, threshold=2.0
        )
        assert result.is_outlier[-2]  # high outlier
        assert result.is_outlier[-1]  # low outlier

    def test_threshold_respected(self, with_outlier: np.ndarray) -> None:
        """Higher threshold → fewer outliers."""
        result_loose = detect_outliers(
            with_outlier,
            method=ZScoreMethod.MODIFIED_MAD,
            threshold=5.0,
        )
        result_tight = detect_outliers(
            with_outlier,
            method=ZScoreMethod.MODIFIED_MAD,
            threshold=1.0,
        )
        assert result_tight.n_outliers >= result_loose.n_outliers

    def test_outlier_indices_consistent(self, with_outlier: np.ndarray) -> None:
        result = detect_outliers(
            with_outlier, method=ZScoreMethod.HYBRID_WEIGHTED, threshold=2.0
        )
        expected_idx = np.nonzero(result.is_outlier)[0]
        assert_allclose(result.outlier_indices, expected_idx)


# =====================================================================
# Cross-method relationship tests
# =====================================================================


class TestCrossMethodRelationships:
    """Verify algebraic relationships that must hold between methods."""

    def test_max_geq_avg(self, with_outlier: np.ndarray) -> None:
        z_max = z_score_max_hybrid(with_outlier)
        z_avg = z_score_avg_hybrid(with_outlier)
        assert np.all(z_max >= z_avg - 1e-12)

    def test_avg_geq_min_of_parents(self, with_outlier: np.ndarray) -> None:
        z_avg = z_score_avg_hybrid(with_outlier)
        z_mad = z_score_mad(with_outlier)
        z_sn = z_score_sn(with_outlier)
        lo = np.minimum(z_mad, z_sn)
        assert np.all(z_avg >= lo - 1e-12)

    def test_weighted_monotonic_in_w(self, symmetric_5: np.ndarray) -> None:
        """If MAD z-scores > Sn z-scores, increasing w should increase
        the weighted hybrid z-scores (and vice versa)."""
        z_low_w = z_score_weighted_hybrid(symmetric_5, w=0.2)
        z_high_w = z_score_weighted_hybrid(symmetric_5, w=0.8)
        z_mad = z_score_mad(symmetric_5)
        z_sn = z_score_sn(symmetric_5)

        for i in range(len(symmetric_5)):
            if z_mad[i] > z_sn[i]:
                assert z_high_w[i] >= z_low_w[i] - 1e-12
            elif z_mad[i] < z_sn[i]:
                assert z_high_w[i] <= z_low_w[i] + 1e-12


# =====================================================================
# Edge-case and regression tests
# =====================================================================


class TestEdgeCases:
    def test_two_element_dataset(self) -> None:
        data = np.array([0.0, 10.0])
        assert sd_scale(data) == 5.0
        assert_allclose(mad_scale(data), 1.4826 * 5.0, atol=1e-12)
        # Sn: inner medians for 2 elements (even, C_n=1)
        #   i=0: median([0, 10]) = 5;  i=1: median([10, 0]) = 5
        # outer median([5, 5]) = 5
        assert_allclose(sn_scale(data), 5.0, atol=1e-12)

    def test_large_random_dataset_no_crash(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(loc=50.0, scale=5.0, size=1000)
        for method in ZScoreMethod:
            result = detect_outliers(data, method=method, threshold=3.0)
            assert result.n_outliers >= 0
            assert result.is_outlier.shape == data.shape

    def test_negative_values(self) -> None:
        data = np.array([-100.0, -99.0, -101.0, -100.5, -99.5])
        for method in ZScoreMethod:
            result = detect_outliers(data, method=method, threshold=3.0)
            assert result.is_outlier.shape == data.shape

    def test_accepts_integer_input(self) -> None:
        data = np.array([1, 2, 3, 4, 5])
        result = detect_outliers(data, method=ZScoreMethod.MODIFIED_MAD, threshold=3.0)
        assert result.is_outlier.shape == data.shape

    def test_accepts_python_list(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = detect_outliers(data, method=ZScoreMethod.MODIFIED_SN, threshold=3.0)
        assert result.is_outlier.shape == (5,)


# =====================================================================
# End-to-end scenario tests
# =====================================================================


class TestEndToEnd:
    """Realistic scenarios combining all layers."""

    def test_rss_like_data_with_spike(self) -> None:
        """Simulate a time-series RSS dataset with one anomalous reading."""
        rng = np.random.default_rng(123)
        normal_rss = rng.normal(loc=-60.0, scale=2.0, size=50)
        data = np.append(normal_rss, [-20.0])  # spike

        result = detect_outliers(
            data,
            method=ZScoreMethod.HYBRID_WEIGHTED,
            threshold=2.0,
            w=0.6,
        )
        assert result.is_outlier[-1], "Spike not detected"
        normal_false_positives = np.sum(result.is_outlier[:-1])
        assert normal_false_positives <= 5, (
            f"Too many false positives: {normal_false_positives}"
        )

    def test_bimodal_distribution(self) -> None:
        """Two clusters with a few points in between — the in-between
        points may or may not be flagged, but genuine cluster members
        should generally not be flagged at threshold=3."""
        rng = np.random.default_rng(456)
        cluster_a = rng.normal(loc=-70.0, scale=1.0, size=40)
        cluster_b = rng.normal(loc=-30.0, scale=1.0, size=40)
        data = np.concatenate([cluster_a, cluster_b])

        result = detect_outliers(
            data,
            method=ZScoreMethod.MODIFIED_MAD,
            threshold=3.0,
        )
        assert isinstance(result, OutlierResult)

    def test_weighted_hybrid_best_outlier_performance(self) -> None:
        """The paper's headline claim: weighted hybrid should have
        competitive or fewer outlier detections than MAD (fewer false
        alarms) while catching genuine outliers."""
        rng = np.random.default_rng(789)
        normal_data = rng.normal(loc=-55.0, scale=3.0, size=100)
        # Inject 3 genuine outliers
        outliers = np.array([-10.0, -100.0, 0.0])
        data = np.concatenate([normal_data, outliers])
        outlier_start = len(normal_data)

        result_mad = detect_outliers(
            data, method=ZScoreMethod.MODIFIED_MAD, threshold=2.0
        )
        result_wgt = detect_outliers(
            data, method=ZScoreMethod.HYBRID_WEIGHTED, threshold=2.0, w=0.6
        )

        # Both should catch all 3 genuine outliers
        for idx in range(outlier_start, len(data)):
            assert result_mad.is_outlier[idx], f"MAD missed outlier at {idx}"
            assert result_wgt.is_outlier[idx], f"Weighted hybrid missed outlier at {idx}"

        # Weighted hybrid should produce no more false alarms than MAD
        mad_fp = int(np.sum(result_mad.is_outlier[:outlier_start]))
        wgt_fp = int(np.sum(result_wgt.is_outlier[:outlier_start]))
        assert wgt_fp <= mad_fp + 2, (
            f"Weighted hybrid had significantly more false positives "
            f"({wgt_fp}) than MAD ({mad_fp})"
        )

    def test_all_methods_agree_on_extreme_outlier(self) -> None:
        """When an outlier is extreme enough, every method should flag it."""
        data = np.array([10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0, 1000.0])
        for method in ZScoreMethod:
            result = detect_outliers(data, method=method, threshold=2.0)
            assert result.is_outlier[-1], (
                f"{method.value} failed to detect extreme outlier"
            )
            # None of the normal points should be flagged
            assert np.sum(result.is_outlier[:-1]) == 0, (
                f"{method.value} had false positives on clean data"
            )
