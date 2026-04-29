# Modified Z-Score Methods with Hybrid Scale Estimators

Implementation of the outlier detection methods described in:

> Yaro, A. S., Maly, F., Prazak, P., & Malý, K. (2024).
> **Outlier Detection Performance of a Modified Z-Score Method in Time-Series RSS Observation With Hybrid Scale Estimators.**
> *IEEE Access*, 12, 12785–12796.
> [doi:10.1109/ACCESS.2024.3356731](https://doi.org/10.1109/ACCESS.2024.3356731)

## Source paper

The paper evaluates how different *scale estimators* affect the outlier detection performance of the modified Z-score (mZ-score) method on time-series Received Signal Strength (RSS) observations. Two established scale estimators — MAD and Sn — are hybridised via three strategies (weighted, maximum, average), and the resulting detectors are benchmarked against MOD, k-means clustering, and DBSCAN on three publicly available RSS datasets. The weighted hybrid is reported as the best-performing variant, achieving low false-alarm and false-negative rates across all datasets.

## What is implemented

The module `z_score_methods.py` provides a pure-NumPy, dependency-light implementation of every Z-score–family method discussed in the paper. Each function maps directly to a numbered equation in the source.

## Validation note

The implementation was validated against the source paper equation by equation. The MAD-based method, the hybrid constructions (`weighted`, `max`, `avg`), and the thresholding logic are structurally consistent with Eqs. 3–4 and 7–11 of Yaro et al. (2024). The only deliberate deviation concerns the `Sn` scale estimator: this module keeps the original Rousseeuw-Croux-style `Sn` definition used in this work, where the inner median is computed over all pairwise absolute differences for a reference observation, including the self-comparison term `|x_i - x_i| = 0`.

Consequently, the implementation should be understood as follows: the hybrid modified Z-score framework follows Yaro et al. (2024), but it is instantiated with the `Sn` definition adopted in this repository rather than with a literal reading of the paper's printed Eq. 5 index restriction. The methods are therefore internally consistent and mathematically coherent, but the `Sn` component is not claimed to be a verbatim reproduction of Eq. 5 as printed in the paper.

### Scale estimators

| Function | Paper ref | Description |
|---|---|---|
| `sd_scale(data)` | Eq. 2 | Population standard deviation (baseline). |
| `mad_scale(data, b=1.4826)` | Eq. 3 | Median Absolute Deviation. The constant *b* = 1.4826 makes MAD consistent with SD for Gaussian data. 37 % Gaussian efficiency, 50 % breakdown point. |
| `sn_scale(data)` | Eqs. 5–6 | Sn estimator (median of pairwise-median absolute differences). Finite-sample correction *C_n* applied per Eq. 6. 58 % Gaussian efficiency, 50 % breakdown point. |

### Z-score methods

| Function | Paper ref | Output | Description |
|---|---|---|---|
| `z_score_sd(data)` | Eq. 2 | Signed | Standard Z-score: `(x − μ) / σ_SD` |
| `z_score_mad(data)` | Eq. 4 | Non-negative | Modified Z-score: `|x − median| / σ_MAD` |
| `z_score_sn(data)` | Eq. 7 | Non-negative | Modified Z-score: `|x − median| / σ_Sn` |

### Hybrid Z-score methods

| Function | Paper ref | Description |
|---|---|---|
| `z_score_weighted_hybrid(data, w)` | Eq. 8 | `w · Z_MAD + (1−w) · Z_Sn`. At *w* = 0 → pure Sn; at *w* = 1 → pure MAD; at *w* = 0.5 → average hybrid. The paper recommends tuning *w* per dataset (optimum values ranged from 0.3 to 0.8 across the three evaluation datasets). |
| `z_score_max_hybrid(data)` | Eq. 9 | Element-wise `max(Z_MAD, Z_Sn)`. Inherits whichever parent flags a larger deviation; empirically behaves like MAD. |
| `z_score_avg_hybrid(data)` | Eq. 10 | `0.5 · (Z_MAD + Z_Sn)`. Mathematically equivalent to the weighted hybrid at *w* = 0.5. |

### Outlier detection

| Function | Paper ref | Description |
|---|---|---|
| `detect_outliers(data, method, threshold, w)` | Eq. 11 | Applies any of the six methods above, then flags observations whose Z-score exceeds the threshold *γ*. Returns an `OutlierResult` dataclass with the boolean mask, raw Z-scores, outlier count, and outlier indices. Default threshold is ±2.0, following the paper's recommended value. |

## Key relationships between methods

These algebraic identities hold by construction and are verified in the test suite:

- `z_score_weighted_hybrid(data, w=0.0)` ≡ `z_score_sn(data)`
- `z_score_weighted_hybrid(data, w=1.0)` ≡ `z_score_mad(data)`
- `z_score_weighted_hybrid(data, w=0.5)` ≡ `z_score_avg_hybrid(data)`
- `z_score_max_hybrid(data)` ≥ `z_score_avg_hybrid(data)` (element-wise)
- `z_score_avg_hybrid(data)` lies between `z_score_mad(data)` and `z_score_sn(data)` (element-wise)

## Usage

```python
import numpy as np
from src.anomaly.z_score_methods import detect_outliers, ZScoreMethod

data = np.array([10.0, 11.0, 10.5, 10.2, 9.8, 10.1, 10.3, 50.0])

result = detect_outliers(
    data,
    method=ZScoreMethod.HYBRID_WEIGHTED,
    threshold=2.0,
    w=0.6,
)

print(result.n_outliers)        # 1
print(result.outlier_indices)   # [7]
print(result.is_outlier)        # [False False False False False False False True]
```

Individual scale estimators and Z-score functions can also be called directly:

```python
from src.anomaly.z_score_methods import mad_scale, sn_scale, z_score_mad

sigma_mad = mad_scale(data)         # MAD-based scale value
sigma_sn  = sn_scale(data)          # Sn-based scale value
z_scores  = z_score_mad(data)       # per-observation modified Z-scores
```

## Tests

54 tests in `tests/test_z_score_methods.py`, organised as:

| Group | Count | What is validated |
|---|---|---|
| Scale estimators | 12 | Hand-computed expected values for `[1,2,3,4,5]`, even-length arrays, constants, Sn correction factors. |
| Z-score computations | 12 | Known values, sign/non-negativity, symmetry, `ValueError` on zero-scale input. |
| Hybrid methods | 9 | Algebraic identities (w=0 → Sn, w=1 → MAD, w=0.5 → avg), max ≥ both parents, avg between parents, weight monotonicity. |
| Outlier detection | 6 | Result structure, all methods catch obvious outliers, threshold monotonicity, two-sided SD detection, index consistency. |
| Cross-method relationships | 3 | max ≥ avg, avg ≥ min(parents), weighted monotonic in *w*. |
| Edge cases | 5 | 2-element arrays, 1000-element random data, negative values, integer and list inputs. |
| End-to-end scenarios | 4 | RSS-like spike, bimodal distribution, weighted hybrid vs MAD false-positive comparison, unanimous extreme-outlier detection. |

Run with:

```bash
cd src
python -m pytest tests/test_z_score_methods.py -v
```

## Implementation notes

- **NumPy only** — no dependency on the rest of the pipeline (no BigQuery, no FastAPI). The module can be used standalone for research experiments.
- **Sn computation** — the inner median includes the self-comparison (`|x_i − x_i| = 0`), consistent with the original Rousseeuw & Croux (1993) definition. The finite-sample correction factor follows the simplified form from Eq. 6 of the paper.
- **Standard Z-score** — uses population SD (`ddof=0`) since the full time series is available at detection time, matching the paper's formulation.
- **Threshold default** — the paper evaluates thresholds ±1, ±2, and ±3 and identifies ±2 as optimal (Sakai et al., 2015; Tukey, 1977). The `detect_outliers` function defaults to 2.0 accordingly.
