"""Centralized configuration for ML model tuning parameters.

This module provides a single source of truth for optimal hyperparameters
discovered through experimentation, including minimum history values for
each model type and data splitting parameters.

Usage:
    from src.tuning_config import get_tuning_config, TuningConfig
    
    # Get the configuration (cached singleton)
    config = get_tuning_config()
    
    # Access minimum history for a specific model
    min_history = config.get_min_history("autoencoder")  # Returns 4
    min_history = config.get_min_history("isolation_forest")  # Returns 5
    min_history = config.get_min_history("statistical")  # Returns 3
    
    # Access data splitting parameters
    test_size = config.data_splitting.test_size  # Returns 0.2
    test_split_amount_of_prices = config.data_splitting.test_split_amount_of_prices  # Returns 2
    random_state = config.data_splitting.random_state  # Returns 42

Configuration File:
    configs/tuning_config.json - Edit this file to update optimal values

Optimal Values (verified via experiments):
    - Autoencoder: minimum_history = 4
    - Isolation Forest: minimum_history = 5
    - Statistical Methods: minimum_history = 3 (default MIN_OBSERVATIONS)
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the configuration file (relative to project root)
CONFIG_FILE_PATH = Path(__file__).parent.parent / "configs" / "tuning_config.json"


@dataclass(frozen=True)
class MinimumHistoryConfig:
    """Optimal minimum history values for each model type.

    Attributes:
        autoencoder: Minimum observations for autoencoder models (default: 4).
        isolation_forest: Minimum observations for isolation forest models (default: 5).
        statistical: Minimum observations for statistical methods (default: 3).
    """

    autoencoder: int = 4
    isolation_forest: int = 5
    statistical: int = 3


@dataclass(frozen=True)
class DataSplittingConfig:
    """Configuration for train/test data splitting.

    Attributes:
        test_size: Fraction of data to use for testing (default: 0.2).
        test_split_amount_of_prices: Number of last observations per product to include in test set (default: 2).
        random_state: Random seed for reproducibility (default: 42).
    """

    test_size: float = 0.2
    test_split_amount_of_prices: int = 2
    random_state: int = 42


@dataclass(frozen=True)
class TuningConfig:
    """Complete tuning configuration.

    Attributes:
        minimum_history: Optimal minimum history values per model type.
        data_splitting: Train/test splitting parameters.
    """

    minimum_history: MinimumHistoryConfig
    data_splitting: DataSplittingConfig

    def get_min_history(self, model_type: str) -> int:
        """Get minimum history value for a specific model type.

        Args:
            model_type: Model type identifier. Supported values:
                - "autoencoder" or "ae"
                - "isolation_forest" or "iforest" or "if"
                - "statistical" or "stat" or "zscore" or "iqr" or "threshold"

        Returns:
            Optimal minimum history value for the model.

        Raises:
            ValueError: If model_type is not recognized.
        """
        model_type_lower = model_type.lower()

        # Map aliases to canonical names
        if model_type_lower in ("autoencoder", "ae"):
            return self.minimum_history.autoencoder
        elif model_type_lower in ("isolation_forest", "iforest", "if"):
            return self.minimum_history.isolation_forest
        elif model_type_lower in ("statistical", "stat", "zscore", "iqr", "threshold"):
            return self.minimum_history.statistical
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Valid types: autoencoder, isolation_forest, statistical"
            )

    def get_all_min_history(self) -> dict[str, int]:
        """Get all minimum history values as a dictionary.

        Returns:
            Dict mapping model type to minimum history value.
        """
        return {
            "autoencoder": self.minimum_history.autoencoder,
            "isolation_forest": self.minimum_history.isolation_forest,
            "statistical": self.minimum_history.statistical,
        }


def _load_config_from_file(path: Path) -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_config(data: dict[str, Any]) -> TuningConfig:
    """Parse raw configuration dict into TuningConfig object.

    Args:
        data: Raw configuration dictionary from JSON.

    Returns:
        Validated TuningConfig object.
    """
    # Parse minimum_history section
    min_history_data = data.get("minimum_history", {})
    minimum_history = MinimumHistoryConfig(
        autoencoder=min_history_data.get("autoencoder", 4),
        isolation_forest=min_history_data.get("isolation_forest", 5),
        statistical=min_history_data.get("statistical", 3),
    )

    # Parse data_splitting section
    splitting_data = data.get("data_splitting", {})
    data_splitting = DataSplittingConfig(
        test_size=splitting_data.get("test_size", 0.2),
        test_split_amount_of_prices=splitting_data.get("test_split_amount_of_prices", 2),
        random_state=splitting_data.get("random_state", 42),
    )

    return TuningConfig(
        minimum_history=minimum_history,
        data_splitting=data_splitting,
    )


@lru_cache(maxsize=1)
def get_tuning_config(config_path: str | None = None) -> TuningConfig:
    """Get tuning configuration from file.

    Loads and caches the configuration from configs/tuning_config.json.
    The config is cached after first call (singleton pattern).

    Args:
        config_path: Optional path to config file. Defaults to
            configs/tuning_config.json in the project root.

    Returns:
        TuningConfig object with all tuning parameters.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    path = Path(config_path) if config_path else CONFIG_FILE_PATH

    try:
        data = _load_config_from_file(path)
        config = _parse_config(data)

        logger.info(
            "tuning_config_loaded",
            extra={
                "config_path": str(path),
                "minimum_history": config.get_all_min_history(),
                "data_splitting": {
                    "test_size": config.data_splitting.test_size,
                    "test_split_amount_of_prices": config.data_splitting.test_split_amount_of_prices,
                    "random_state": config.data_splitting.random_state,
                },
            },
        )

        return config

    except FileNotFoundError:
        logger.warning(
            "tuning_config_not_found_using_defaults",
            extra={"config_path": str(path)},
        )
        # Return default config if file not found
        return TuningConfig(
            minimum_history=MinimumHistoryConfig(),
            data_splitting=DataSplittingConfig(),
        )

    except json.JSONDecodeError as e:
        logger.error(
            "tuning_config_parse_error",
            extra={"config_path": str(path), "error": str(e)},
        )
        raise


def reset_tuning_config() -> None:
    """Reset the cached tuning config (useful for testing).

    After calling this, the next get_tuning_config() call will re-read
    the configuration file.
    """
    get_tuning_config.cache_clear()


# Convenience function for quick access to minimum history
def get_min_history(model_type: str) -> int:
    """Get minimum history for a model type (convenience function).

    Args:
        model_type: Model type identifier (autoencoder, isolation_forest, statistical).

    Returns:
        Optimal minimum history value.
    """
    return get_tuning_config().get_min_history(model_type)
