"""Machine Learning Anomaly Detection Module.

Purpose:
    Provide ML-based anomaly detection methods for systematic comparison
    with statistical baselines. These represent the thesis contribution.

Key Components:
    - isolation_forest.py: Isolation Forest for tabular features
    - eif.py: Extended Isolation Forest for tabular features
    - autoencoder.py: Autoencoder for learning normal patterns
    - rrcf.py: Robust Random Cut Forest core plus detector wrapper
    - embedding_drift.py: Text embedding drift detection (planned)

Dependencies:
    - scikit-learn (Isolation Forest)
    - torch (Autoencoder)
    - numpy (RRCF)
    - sentence-transformers (embeddings, planned)
    - src/features/ (feature extraction)

Consumed by:
    - src/services/refinery.py
    - src/evaluation/ (method comparison)

Module TODOs:
    - [x] Implement embedding drift detection (ContentDriftDetector in features/embeddings.py)
    - [x] Add model save/load functionality for production use
    - [x] Implement RRCF for streaming anomaly detection
    - [x] Tune Isolation Forest thresholds (scripts/tune_isolation_forest.py)
    - [x] Tune Autoencoder thresholds (scripts/tune_autoencoder.py)
    - [ ] Add SHAP explanations for Isolation Forest
"""


def validate_feature_schema(
    actual_names: list[str],
    expected_names: list[str],
    detector_name: str,
) -> None:
    """Validate that extracted features match model's expected schema.

    Catches configuration drift or feature extraction bugs between training
    and inference. Should be called at the start of detect_batch() methods.

    Args:
        actual_names: Feature names from current extraction.
        expected_names: Feature names saved during training.
        detector_name: Name of detector for error messages.

    Raises:
        ValueError: If feature count or names don't match.
    """
    if len(actual_names) != len(expected_names):
        raise ValueError(
            f"{detector_name} feature count mismatch: "
            f"expected {len(expected_names)}, got {len(actual_names)}. "
            f"Expected: {expected_names}, Got: {actual_names}"
        )

    if actual_names != expected_names:
        # Find which features differ
        missing = set(expected_names) - set(actual_names)
        extra = set(actual_names) - set(expected_names)
        raise ValueError(
            f"{detector_name} feature name mismatch. "
            f"Missing: {missing}, Extra: {extra}"
        )


from src.anomaly.ml.isolation_forest import (
    IsolationForestDetector,
    IsolationForestConfig,
)
from src.anomaly.ml.eif import (
    EIFConfig,
    EIFDetector,
)
from src.anomaly.ml.autoencoder import (
    AutoencoderDetector,
    AutoencoderConfig,
)
from src.anomaly.ml.rrcf import (
    RRCF,
    RRCFDetector,
    RRCFDetectorConfig,
    RRCFResult,
)
__all__ = [
    # Validation
    "validate_feature_schema",
    # Isolation Forest
    "IsolationForestDetector",
    "IsolationForestConfig",
    # Extended Isolation Forest
    "EIFDetector",
    "EIFConfig",
    # Autoencoder
    "AutoencoderDetector",
    "AutoencoderConfig",
    # RRCF
    "RRCF",
    "RRCFDetector",
    "RRCFDetectorConfig",
    "RRCFResult",
]
