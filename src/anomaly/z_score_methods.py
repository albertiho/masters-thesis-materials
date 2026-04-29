"""Modified Z-Score Methods with Hybrid Scale Estimators.

Implements the outlier detection methods from:
    Yaro, A. S., Maly, F., Prazak, P., & Malý, K. (2024).
    "Outlier Detection Performance of a Modified Z-Score Method in
     Time-Series RSS Observation With Hybrid Scale Estimators."
    IEEE Access, 12, 12785-12796. doi:10.1109/ACCESS.2024.3356731

Methods implemented:
    - Standard Z-score with SD scale estimator (eq. 2)
    - Modified Z-score with MAD scale estimator (eq. 3-4)
    - Modified Z-score with Sn scale estimator (eq. 5-7)
    - Modified Z-score with Weighted Hybrid scale estimator (eq. 8)
    - Modified Z-score with Maximum Hybrid scale estimator (eq. 9)
    - Modified Z-score with Average Hybrid scale estimator (eq. 10)
    - Outlier detection via threshold comparison (eq. 11)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ZScoreMethod(Enum):
    """Available Z-score computation methods."""

    STANDARD_SD = "standard_sd"
    MODIFIED_MAD = "modified_mad"
    MODIFIED_SN = "modified_sn"
    HYBRID_WEIGHTED = "hybrid_weighted"
    HYBRID_MAX = "hybrid_max"
    HYBRID_AVG = "hybrid_avg"


# ---------------------------------------------------------------------------
# Scale estimators
# ---------------------------------------------------------------------------


def sd_scale(data: NDArray[np.floating]) -> float:
    """Standard deviation scale estimator.

    Uses population standard deviation (N denominator), matching the
    conventional Z-score formulation where the full dataset is available.

    Args:
        data: 1-D array of observations.

    Returns:
        Population standard deviation of *data*.
    """
    return float(np.std(data, ddof=0))


def mad_scale(data: NDArray[np.floating], b: float = 1.4826) -> float:
    """Median Absolute Deviation (MAD) scale estimator (eq. 3).

    σ_MAD = b × median(|x_i − median(X)|)

    The constant *b* = 1.4826 makes the MAD consistent with the standard
    deviation for normally distributed data.  The MAD has 37 % Gaussian
    efficiency and a 50 % breakdown point.

    Args:
        data: 1-D array of observations.
        b: Scaling constant (default 1.4826 for normal consistency).

    Returns:
        MAD-based scale value.
    """
    median = float(np.median(data))
    return b * float(np.median(np.abs(data - median)))


def _sn_correction_factor(n: int) -> float:
    """Finite-sample correction factor C_n for the Sn estimator (eq. 6).

    C_n = n / (n − 0.9)   for odd n
    C_n = 1                for even n

    These values are valid for n > 9.  For very small samples the original
    Rousseeuw & Croux (1992) tabulated values would be more accurate, but
    the paper under implementation uses this simplified form.

    Args:
        n: Sample size.

    Returns:
        Finite-sample correction factor.
    """
    if n % 2 == 1:
        return n / (n - 0.9)
    return 1.0


def sn_scale(data: NDArray[np.floating]) -> float:
    """Sn scale estimator (eq. 5).

    σ_Sn = C_n × median_i { median_j { |x_i − x_j| } }

    For each observation i the inner median is computed over *all* j
    (including j = i, which contributes 0), consistent with Rousseeuw &
    Croux (1993).  The Sn estimator has 58 % Gaussian efficiency and a
    50 % breakdown point.

    Args:
        data: 1-D array of observations.

    Returns:
        Sn-based scale value.
    """
    n = len(data)
    c_n = _sn_correction_factor(n)

    inner_medians = np.empty(n)
    for i in range(n):
        inner_medians[i] = np.median(np.abs(data[i] - data))

    return c_n * float(np.median(inner_medians))


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------


def z_score_sd(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Standard Z-score using SD scale estimator (eq. 2).

    Z_SD(x_n) = (x_n − μ) / σ_SD

    Returns signed Z-scores (negative = below mean, positive = above mean).

    Args:
        data: 1-D array of observations.

    Returns:
        Array of signed Z-scores.

    Raises:
        ValueError: If the standard deviation is zero.
    """
    sigma = sd_scale(data)
    if sigma == 0.0:
        raise ValueError("Standard deviation is zero; Z-score is undefined.")
    mean = float(np.mean(data))
    return (data - mean) / sigma


def z_score_mad(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Modified Z-score using MAD scale estimator (eq. 4).

    Z_MAD(x_n) = |x_n − median(X)| / σ_MAD

    Returns non-negative Z-scores (absolute deviation from median).

    Args:
        data: 1-D array of observations.

    Returns:
        Array of non-negative modified Z-scores.

    Raises:
        ValueError: If the MAD scale is zero.
    """
    sigma = mad_scale(data)
    if sigma == 0.0:
        raise ValueError("MAD scale is zero; modified Z-score is undefined.")
    median = float(np.median(data))
    return np.abs(data - median) / sigma


def z_score_sn(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Modified Z-score using Sn scale estimator (eq. 7).

    Z_Sn(x_n) = |x_n − median(X)| / σ_Sn

    Returns non-negative Z-scores (absolute deviation from median).

    Args:
        data: 1-D array of observations.

    Returns:
        Array of non-negative modified Z-scores.

    Raises:
        ValueError: If the Sn scale is zero.
    """
    sigma = sn_scale(data)
    if sigma == 0.0:
        raise ValueError("Sn scale is zero; modified Z-score is undefined.")
    median = float(np.median(data))
    return np.abs(data - median) / sigma


# ---------------------------------------------------------------------------
# Hybrid Z-score methods
# ---------------------------------------------------------------------------


def z_score_weighted_hybrid(
    data: NDArray[np.floating],
    w: float = 0.5,
) -> NDArray[np.floating]:
    """Weighted hybrid Z-score combining MAD and Sn (eq. 8).

    Z_wgt(x_n) = w × Z_MAD(x_n) + (1 − w) × Z_Sn(x_n)

    At w = 0 the estimator reduces to pure Sn; at w = 1 it reduces to pure
    MAD; at w = 0.5 it is equivalent to the average hybrid.

    Args:
        data: 1-D array of observations.
        w: Weight assigned to the MAD component (0 ≤ w ≤ 1).

    Returns:
        Array of weighted hybrid Z-scores.

    Raises:
        ValueError: If *w* is outside [0, 1].
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"Weight w must be in [0, 1], got {w}.")
    z_mad = z_score_mad(data)
    z_sn = z_score_sn(data)
    return w * z_mad + (1.0 - w) * z_sn


def z_score_max_hybrid(
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Maximum hybrid Z-score (eq. 9).

    Z_max(x_n) = max(Z_MAD(x_n), Z_Sn(x_n))

    Takes the element-wise maximum of MAD and Sn Z-scores, thereby
    inheriting whichever estimator flags a larger deviation.

    Args:
        data: 1-D array of observations.

    Returns:
        Array of maximum hybrid Z-scores.
    """
    z_mad = z_score_mad(data)
    z_sn = z_score_sn(data)
    return np.maximum(z_mad, z_sn)


def z_score_avg_hybrid(
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Average hybrid Z-score (eq. 10).

    Z_avg(x_n) = 0.5 × (Z_MAD(x_n) + Z_Sn(x_n))

    Equivalent to the weighted hybrid at w = 0.5.

    Args:
        data: 1-D array of observations.

    Returns:
        Array of average hybrid Z-scores.
    """
    z_mad = z_score_mad(data)
    z_sn = z_score_sn(data)
    return 0.5 * (z_mad + z_sn)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


@dataclass
class OutlierResult:
    """Result container for outlier detection.

    Attributes:
        is_outlier: Boolean mask (True = outlier).
        z_scores: The Z-score array used for the decision.
        threshold: Threshold that was applied.
        method: Which Z-score method produced these scores.
        n_outliers: Count of outliers detected.
        outlier_indices: Integer indices of detected outliers.
    """

    is_outlier: NDArray[np.bool_]
    z_scores: NDArray[np.floating]
    threshold: float
    method: ZScoreMethod
    n_outliers: int
    outlier_indices: NDArray[np.intp]


def detect_outliers(
    data: NDArray[np.floating],
    method: ZScoreMethod = ZScoreMethod.HYBRID_WEIGHTED,
    threshold: float = 2.0,
    w: float = 0.5,
) -> OutlierResult:
    """Detect outliers in a 1-D time series using the specified Z-score method.

    For the standard Z-score (signed), an observation is an outlier when
    |Z| > threshold.  For all modified / hybrid methods (which already
    return absolute Z-scores), the condition is Z > threshold (eq. 11).

    Args:
        data: 1-D array of observations.
        method: Z-score variant to use (see :class:`ZScoreMethod`).
        threshold: Z-score threshold γ for outlier flagging (default 2.0,
            following the paper's recommended value).
        w: Weight for the weighted hybrid method (ignored by other methods).

    Returns:
        :class:`OutlierResult` with detection outcomes.
    """
    data = np.asarray(data, dtype=np.float64)

    z_func_map = {
        ZScoreMethod.STANDARD_SD: lambda d: z_score_sd(d),
        ZScoreMethod.MODIFIED_MAD: lambda d: z_score_mad(d),
        ZScoreMethod.MODIFIED_SN: lambda d: z_score_sn(d),
        ZScoreMethod.HYBRID_WEIGHTED: lambda d: z_score_weighted_hybrid(d, w),
        ZScoreMethod.HYBRID_MAX: lambda d: z_score_max_hybrid(d),
        ZScoreMethod.HYBRID_AVG: lambda d: z_score_avg_hybrid(d),
    }

    z_scores = z_func_map[method](data)

    if method == ZScoreMethod.STANDARD_SD:
        is_outlier = np.abs(z_scores) > threshold
    else:
        is_outlier = z_scores > threshold

    outlier_indices = np.nonzero(is_outlier)[0]

    return OutlierResult(
        is_outlier=is_outlier,
        z_scores=z_scores,
        threshold=threshold,
        method=method,
        n_outliers=int(np.sum(is_outlier)),
        outlier_indices=outlier_indices,
    )
