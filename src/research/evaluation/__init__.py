"""Evaluation Module for Anomaly Detection Methods.

Purpose:
    Evaluate and compare different anomaly detection methods using
    synthetic anomaly injection (since we have no labeled data).
    
    This module is IN-SCOPE for the thesis project but is NOT part of
    the production pipeline. It supports research experimentation and
    method comparison.

Key Components:
    - detector_evaluator.py: DetectorEvaluator wraps detector with isolated cache
    - test_orchestrator.py: TestOrchestrator coordinates parallel detector comparison
    - synthetic.py: Inject synthetic anomalies into real data

Architecture:
    DetectorEvaluator: Wraps a single detector with its own isolated cache.
        Each evaluator maintains independent baseline statistics.
    
    TestOrchestrator: Manages multiple DetectorEvaluators with parallel execution.
        Coordinates clearing, populating, and processing across evaluators.

Evaluation Strategy:
    1. Take real "normal" data
    2. Inject known synthetic anomalies (price, content, scraper bugs)
    3. Create DetectorEvaluator for each detector
    4. Use TestOrchestrator.run_comparison() for parallel evaluation
    5. Compare detection rates and false positives

Dependencies:
    - src/features/ (feature extraction)
    - src/anomaly/ (detection methods)
    - numpy, pandas

Consumed by:
    - Thesis evaluation (research/training/scripts/compare_detectors.py)
    - Method comparison experiments (research/training/scripts/analyze_*.py)
    - Model training scripts (research/training/scripts/train_*.py)
    - Hyperparameter tuning (research/training/scripts/grid_search_*.py)

Module TODOs:
    - [ ] Add visualization helpers for thesis plots
    - [ ] Implement cross-validation for threshold tuning
    - [ ] Add expert review workflow for sample anomalies
"""

from src.research.evaluation.detector_evaluator import DetectorEvaluator
from src.research.evaluation.test_orchestrator import (
    ComparisonResult,
    DetectorMetrics,
    TestOrchestrator,
    create_expanded_statistical_evaluators,
    create_statistical_evaluators,
)
from src.research.evaluation.synthetic import (
    AnomalyInjectionConfig,
    InjectedAnomaly,
    PRODUCTION_ANOMALY_TYPES,
    SyntheticAnomalyInjector,
    SyntheticAnomalyType,
    evaluate_classifier,
    evaluate_detection,
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
