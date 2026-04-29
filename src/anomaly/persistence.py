"""Local model persistence for the standalone thesis repository.

All saved artifacts live on the local filesystem under `artifacts/models/` by
default. The standalone `src/` package does not support remote persistence.
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import get_config

if TYPE_CHECKING:
    from src.anomaly.ml.autoencoder import AutoencoderDetector
    from src.anomaly.ml.eif import EIFDetector
    from src.anomaly.ml.isolation_forest import IsolationForestDetector
    from src.anomaly.ml.rrcf import RRCFDetector

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL_ROOT = "artifacts/models"


def _resolve_repo_root() -> Path:
    """Return the repository root for the standalone training package."""
    return Path(__file__).resolve().parents[2]


@dataclass
class StatisticalConfig:
    """Configuration for statistical anomaly detectors."""

    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    price_change_threshold: float = 0.20
    zscore_min_history: int = 3
    iqr_min_history: int = 3
    threshold_min_history: int = 2
    tuned_at: datetime | None = None
    tuned_from_data: str | None = None
    version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "zscore_threshold": self.zscore_threshold,
            "iqr_multiplier": self.iqr_multiplier,
            "price_change_threshold": self.price_change_threshold,
            "zscore_min_history": self.zscore_min_history,
            "iqr_min_history": self.iqr_min_history,
            "threshold_min_history": self.threshold_min_history,
            "tuned_at": self.tuned_at.isoformat() if self.tuned_at else None,
            "tuned_from_data": self.tuned_from_data,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatisticalConfig:
        """Create an instance from a dictionary."""
        payload = data.copy()
        if payload.get("tuned_at") and isinstance(payload["tuned_at"], str):
            payload["tuned_at"] = datetime.fromisoformat(payload["tuned_at"])
        return cls(
            zscore_threshold=payload.get("zscore_threshold", 3.0),
            iqr_multiplier=payload.get("iqr_multiplier", 1.5),
            price_change_threshold=payload.get("price_change_threshold", 0.20),
            zscore_min_history=payload.get("zscore_min_history", 3),
            iqr_min_history=payload.get("iqr_min_history", 3),
            threshold_min_history=payload.get("threshold_min_history", 2),
            tuned_at=payload.get("tuned_at"),
            tuned_from_data=payload.get("tuned_from_data"),
            version=payload.get("version", "2.0"),
        )


@dataclass
class ModelMetadata:
    """Metadata for a saved model."""

    model_name: str
    competitor: str
    environment: str
    trained_at: datetime
    n_samples: int
    feature_names: list[str]
    config: dict[str, Any]
    version: str = "1.0"
    storage_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        payload = asdict(self)
        payload["trained_at"] = self.trained_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create an instance from a dictionary."""
        payload = data.copy()
        if isinstance(payload.get("trained_at"), str):
            payload["trained_at"] = datetime.fromisoformat(payload["trained_at"])

        return cls(**payload)


class ModelPersistence:
    """Handle saving and loading models from local storage."""

    def __init__(
        self,
        model_root: str | Path | None = None,
    ) -> None:
        """Initialize local persistence.

        Args:
            model_root: Root directory for local artifacts. Defaults to
                `MODEL_ROOT` or `artifacts/models` relative to the repo root.
        """
        cfg = get_config()
        self.environment = cfg.environment.value

        local_root_value = Path(model_root or os.environ.get("MODEL_ROOT", DEFAULT_LOCAL_MODEL_ROOT))
        if not local_root_value.is_absolute():
            local_root_value = (_resolve_repo_root() / local_root_value).resolve()
        self.model_root = local_root_value
        self.model_root.mkdir(parents=True, exist_ok=True)

    @property
    def models_root_description(self) -> str:
        """Return the active model root for user-facing logs."""
        return str((self.model_root / self.environment).resolve())

    def _models_base_prefix(self) -> str:
        return self.environment

    def _absolute_local_path(self, relative_path: str) -> Path:
        return self.model_root / Path(relative_path)

    def _storage_uri(self, relative_path: str) -> str:
        return str(self._absolute_local_path(relative_path).resolve())

    def _path_exists(self, relative_path: str) -> bool:
        return self._absolute_local_path(relative_path).exists()

    def _list_relative_paths(self, prefix: str, suffix: str) -> list[str]:
        root = self._absolute_local_path(prefix)
        if not root.exists():
            return []
        return sorted(
            path.relative_to(self.model_root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.as_posix().endswith(suffix)
        )

    def _get_model_prefix(self, competitor: str, model_name: str) -> str:
        return f"{self._models_base_prefix()}/{competitor}/{model_name}/"

    def _get_model_path(self, competitor: str, model_name: str, filename: str) -> str:
        return f"{self._get_model_prefix(competitor, model_name)}{filename}"

    def _upload_bytes(self, data: bytes, storage_path: str) -> str:
        """Persist bytes to local storage and return the absolute path."""
        absolute_path = self._absolute_local_path(storage_path)
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_bytes(data)

        uri = self._storage_uri(storage_path)
        logger.info("model_uploaded", extra={"storage_path": uri, "size_bytes": len(data)})
        return uri

    def _download_bytes(self, storage_path: str) -> bytes:
        """Load bytes from local storage."""
        absolute_path = self._absolute_local_path(storage_path)
        data = absolute_path.read_bytes()
        logger.info(
            "model_downloaded",
            extra={"storage_path": str(absolute_path.resolve()), "size_bytes": len(data)},
        )
        return data

    def _save_metadata(self, metadata: ModelMetadata, competitor: str, model_name: str) -> str:
        metadata_path = self._get_model_path(competitor, model_name, "metadata.json")
        prefix = self._get_model_prefix(competitor, model_name)
        metadata.storage_path = self._storage_uri(prefix)
        data = json.dumps(metadata.to_dict(), indent=2).encode("utf-8")
        return self._upload_bytes(data, metadata_path)

    def _save_local_joblib_detector(
        self,
        detector: Any,
        competitor: str,
        model_name: str,
        n_samples: int,
    ) -> str:
        if not detector._is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_path = self._get_model_path(competitor, model_name, "model.joblib")
        absolute_model_path = self._absolute_local_path(model_path)
        absolute_model_path.parent.mkdir(parents=True, exist_ok=True)
        detector.save(str(absolute_model_path))

        metadata = ModelMetadata(
            model_name=model_name,
            competitor=competitor,
            environment=self.environment,
            trained_at=datetime.now(timezone.utc),
            n_samples=n_samples,
            feature_names=list(detector._feature_names),
            config=detector.get_model_info().get("config", {}),
        )
        self._save_metadata(metadata, competitor, model_name)

        model_uri = self._storage_uri(self._get_model_prefix(competitor, model_name))
        logger.info(
            "%s_saved",
            model_name,
            extra={
                "competitor": competitor,
                "environment": self.environment,
                "storage_path": model_uri,
                "n_samples": n_samples,
            },
        )
        return model_uri

    def _load_local_joblib_detector(
        self,
        competitor: str,
        model_name: str,
        detector_cls: type[Any],
    ) -> Any:
        model_path = self._get_model_path(competitor, model_name, "model.joblib")
        absolute_model_path = self._absolute_local_path(model_path)
        detector = detector_cls.load(str(absolute_model_path))
        logger.info(
            "%s_loaded",
            model_name,
            extra={
                "competitor": competitor,
                "environment": self.environment,
                "n_features": len(detector._feature_names),
            },
        )
        return detector

    def save_isolation_forest(
        self,
        detector: IsolationForestDetector,
        competitor: str,
        n_samples: int,
    ) -> str:
        """Save a trained Isolation Forest model."""
        return self._save_local_joblib_detector(detector, competitor, "isolation_forest", n_samples)

    def load_isolation_forest(self, competitor: str) -> IsolationForestDetector:
        """Load an Isolation Forest model from local storage."""
        from src.anomaly.ml.isolation_forest import IsolationForestDetector

        return self._load_local_joblib_detector(competitor, "isolation_forest", IsolationForestDetector)

    def save_eif(
        self,
        detector: EIFDetector,
        competitor: str,
        n_samples: int,
    ) -> str:
        """Save a trained EIF model."""
        return self._save_local_joblib_detector(detector, competitor, "eif", n_samples)

    def load_eif(self, competitor: str) -> EIFDetector:
        """Load an EIF model from local storage."""
        from src.anomaly.ml.eif import EIFDetector

        return self._load_local_joblib_detector(competitor, "eif", EIFDetector)

    def save_rrcf(
        self,
        detector: RRCFDetector,
        competitor: str,
        n_samples: int,
    ) -> str:
        """Save a trained RRCF detector."""
        return self._save_local_joblib_detector(detector, competitor, "rrcf", n_samples)

    def load_rrcf(self, competitor: str) -> RRCFDetector:
        """Load an RRCF detector from local storage."""
        from src.anomaly.ml.rrcf import RRCFDetector

        return self._load_local_joblib_detector(competitor, "rrcf", RRCFDetector)

    def save_autoencoder(
        self,
        detector: AutoencoderDetector,
        competitor: str,
        n_samples: int,
    ) -> str:
        """Save a trained Autoencoder model."""
        import torch

        if not detector._is_fitted:
            raise ValueError("Cannot save unfitted model")

        buffer = io.BytesIO()
        torch.save(
            {
                "encoder_state_dict": detector._model.encoder.state_dict(),
                "decoder_state_dict": detector._model.decoder.state_dict(),
                "config": {
                    "input_dim": detector.config.input_dim,
                    "hidden_dims": detector.config.hidden_dims,
                    "latent_dim": detector.config.latent_dim,
                    "learning_rate": detector.config.learning_rate,
                    "epochs": detector.config.epochs,
                    "batch_size": detector.config.batch_size,
                    "anomaly_threshold": detector.config.anomaly_threshold,
                    "dropout": detector.config.dropout,
                    "use_embeddings": detector.config.use_embeddings,
                    "embedding_dim": detector.config.embedding_dim,
                    "embedding_model": detector.config.embedding_model,
                },
                "threshold": detector._threshold,
                "mean": detector._mean.tolist() if detector._mean is not None else None,
                "std": detector._std.tolist() if detector._std is not None else None,
                "feature_names": detector._feature_names,
            },
            buffer,
        )
        model_path = self._get_model_path(competitor, "autoencoder", "model.pt")
        self._upload_bytes(buffer.getvalue(), model_path)

        metadata = ModelMetadata(
            model_name="autoencoder",
            competitor=competitor,
            environment=self.environment,
            trained_at=datetime.now(timezone.utc),
            n_samples=n_samples,
            feature_names=detector._feature_names,
            config={
                "input_dim": detector.config.input_dim,
                "latent_dim": detector.config.latent_dim,
                "epochs": detector.config.epochs,
                "threshold": detector._threshold,
            },
        )
        self._save_metadata(metadata, competitor, "autoencoder")

        model_uri = self._storage_uri(self._get_model_prefix(competitor, "autoencoder"))
        logger.info(
            "autoencoder_saved",
            extra={
                "competitor": competitor,
                "environment": self.environment,
                "storage_path": model_uri,
                "n_samples": n_samples,
            },
        )
        return model_uri

    def load_autoencoder(self, competitor: str) -> AutoencoderDetector:
        """Load an Autoencoder model from local storage."""
        import numpy as np
        import torch

        from src.anomaly.ml.autoencoder import (
            AutoencoderConfig,
            AutoencoderDetector,
            AutoencoderModel,
        )

        model_path = self._get_model_path(competitor, "autoencoder", "model.pt")
        buffer = io.BytesIO(self._download_bytes(model_path))
        saved_data = torch.load(buffer, map_location="cpu")

        config_dict = saved_data["config"]
        use_embeddings = config_dict.get("use_embeddings", False)
        embedding_dim = config_dict.get("embedding_dim", 0)

        if not use_embeddings and saved_data.get("mean") is not None:
            n_features = len(saved_data["mean"])
            if n_features > 50:
                use_embeddings = True
                embedding_dim = n_features - 9

        config = AutoencoderConfig(
            input_dim=config_dict["input_dim"],
            hidden_dims=config_dict["hidden_dims"],
            latent_dim=config_dict["latent_dim"],
            learning_rate=config_dict["learning_rate"],
            epochs=config_dict["epochs"],
            batch_size=config_dict["batch_size"],
            anomaly_threshold=config_dict["anomaly_threshold"],
            dropout=config_dict["dropout"],
            use_embeddings=use_embeddings,
            embedding_dim=embedding_dim,
            embedding_model=config_dict.get("embedding_model", ""),
        )

        detector = AutoencoderDetector(config)
        detector._model = AutoencoderModel(config)
        detector._model.encoder.load_state_dict(saved_data["encoder_state_dict"])
        detector._model.decoder.load_state_dict(saved_data["decoder_state_dict"])
        detector._model.encoder.eval()
        detector._model.decoder.eval()
        detector._threshold = saved_data["threshold"]
        detector._mean = np.array(saved_data["mean"]) if saved_data["mean"] else None
        detector._std = np.array(saved_data["std"]) if saved_data["std"] else None
        detector._feature_names = saved_data["feature_names"]
        detector._is_fitted = True

        logger.info(
            "autoencoder_loaded",
            extra={
                "competitor": competitor,
                "environment": self.environment,
                "input_dim": config.input_dim,
                "threshold": detector._threshold,
            },
        )
        return detector

    def list_models(self, competitor: str | None = None) -> list[ModelMetadata]:
        """List all saved models for the current environment."""
        prefix = f"{self._models_base_prefix()}/"
        if competitor:
            prefix = f"{self._models_base_prefix()}/{competitor}/"

        metadata_files = self._list_relative_paths(prefix, "metadata.json")
        models: list[ModelMetadata] = []

        for relative_path in metadata_files:
            try:
                payload = self._download_bytes(relative_path)
                models.append(ModelMetadata.from_dict(json.loads(payload.decode("utf-8"))))
            except Exception as exc:
                logger.warning(
                    "failed_to_load_metadata",
                    extra={"path": relative_path, "error": str(exc)},
                )

        logger.info(
            "models_listed",
            extra={
                "environment": self.environment,
                "competitor": competitor,
                "count": len(models),
            },
        )
        return models

    def model_exists(self, competitor: str, model_name: str) -> bool:
        """Check if a model exists."""
        return self._path_exists(self._get_model_path(competitor, model_name, "metadata.json"))

    def _get_statistical_config_path(self, name: str) -> str:
        return f"{self._models_base_prefix()}/{name}/statistical/config.json"

    def save_statistical_config(self, config: StatisticalConfig, name: str) -> str:
        """Save a statistical config."""
        config_path = self._get_statistical_config_path(name)
        uri = self._upload_bytes(json.dumps(config.to_dict(), indent=2).encode("utf-8"), config_path)
        logger.info(
            "statistical_config_saved",
            extra={
                "name": name,
                "environment": self.environment,
                "storage_path": uri,
            },
        )
        return uri

    def load_statistical_config(self, name: str) -> StatisticalConfig | None:
        """Load a statistical config if it exists."""
        config_path = self._get_statistical_config_path(name)
        if not self._path_exists(config_path):
            return None
        try:
            data = self._download_bytes(config_path)
            return StatisticalConfig.from_dict(json.loads(data.decode("utf-8")))
        except Exception as exc:
            logger.warning(
                "statistical_config_load_failed",
                extra={"name": name, "environment": self.environment, "error": str(exc)},
            )
            return None

    def load_statistical_config_with_fallback(
        self,
        competitor: str | None = None,
        country_segment: str | None = None,
    ) -> StatisticalConfig | None:
        """Load a statistical config using competitor -> segment -> global fallback."""
        if competitor:
            config = self.load_statistical_config(competitor)
            if config:
                return config
        if country_segment:
            config = self.load_statistical_config(country_segment)
            if config:
                return config
        return self.load_statistical_config("_global")

    def list_statistical_configs(self) -> list[tuple[str, StatisticalConfig]]:
        """List all statistical configs for the current environment."""
        config_files = self._list_relative_paths(f"{self._models_base_prefix()}/", "config.json")
        configs: list[tuple[str, StatisticalConfig]] = []

        for relative_path in config_files:
            if not relative_path.endswith("/statistical/config.json"):
                continue
            parts = relative_path.split("/")
            if len(parts) < 4:
                continue
            name = parts[-3]
            try:
                payload = self._download_bytes(relative_path)
                configs.append((name, StatisticalConfig.from_dict(json.loads(payload.decode("utf-8")))))
            except Exception as exc:
                logger.warning(
                    "failed_to_load_statistical_config",
                    extra={"path": relative_path, "error": str(exc)},
                )

        return configs

    def statistical_config_exists(self, name: str) -> bool:
        """Check if a statistical config exists."""
        return self._path_exists(self._get_statistical_config_path(name))

    def _get_min_history_config_path(self) -> str:
        return f"configs/{self.environment}/min_history_config.json"

    def save_min_history_config(self, config: dict[str, int]) -> str:
        """Save min-history config to local storage."""
        config_path = self._get_min_history_config_path()
        return self._upload_bytes(json.dumps(config, indent=2).encode("utf-8"), config_path)

    def load_min_history_config(self) -> dict[str, int]:
        """Load min-history config from local storage."""
        config_path = self._get_min_history_config_path()
        if not self._path_exists(config_path):
            raise FileNotFoundError(
                f"Min history config not found: {self._storage_uri(config_path)}. "
                "Create it first using save_min_history_config()."
            )
        return json.loads(self._download_bytes(config_path).decode("utf-8"))

    def min_history_config_exists(self) -> bool:
        """Check if min-history config exists."""
        return self._path_exists(self._get_min_history_config_path())
