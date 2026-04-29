"""Research Module - Thesis Evaluation and Experimentation Tools.

Purpose:
    Contains research-focused code for the thesis project. These modules are
    in-scope for the project but are NOT part of the production pipeline.
    They support experimentation, evaluation, and comparison of detection methods.

Key Components:
    - evaluation/: Synthetic anomaly injection and detector comparison framework

Why Separate from Production:
    Research code has different requirements than production code:
    - Experimentation over stability
    - Comprehensive metrics over performance
    - Controlled evaluation environments
    
    Keeping research code separate:
    1. Clarifies what runs in production vs. what's for analysis
    2. Prevents accidental coupling between research experiments and production
    3. Makes it easier to iterate on research without affecting production

Consumed by:
    - research/training/scripts/compare_detectors.py
    - research/training/scripts/compare_granularity_models.py
    - research/training/scripts/train_*.py
    - research/training/scripts/grid_search_*.py
    - research/training/scripts/validate_*.py
    - research/training/scripts/analyze_*.py

Dependencies:
    - src/anomaly/ (detection methods)
    - src/features/ (feature extraction)
    - numpy, pandas, scikit-learn
"""

from src.research.evaluation import (
    # Core evaluation classes
    DetectorEvaluator,
    TestOrchestrator,
    DetectorMetrics,
    ComparisonResult,
    create_expanded_statistical_evaluators,
    create_statistical_evaluators,
    # Synthetic anomaly injection
    SyntheticAnomalyInjector,
    AnomalyInjectionConfig,
    InjectedAnomaly,
    SyntheticAnomalyType,
    PRODUCTION_ANOMALY_TYPES,
    evaluate_detection,
    evaluate_classifier,
    generate_all_anomaly_variants,
    inject_anomalies_to_dataframe,
)

__all__ = [
    # Core evaluation classes
    "DetectorEvaluator",
    "TestOrchestrator",
    "DetectorMetrics",
    "ComparisonResult",
    "create_expanded_statistical_evaluators",
    "create_statistical_evaluators",
    # Synthetic anomaly injection
    "SyntheticAnomalyInjector",
    "AnomalyInjectionConfig",
    "InjectedAnomaly",
    "SyntheticAnomalyType",
    "PRODUCTION_ANOMALY_TYPES",
    "evaluate_detection",
    "evaluate_classifier",
    "generate_all_anomaly_variants",
    "inject_anomalies_to_dataframe",
]
