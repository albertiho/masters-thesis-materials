"""Autoencoder Anomaly Detection.

Autoencoders learn to compress and reconstruct "normal" data.
Anomalies have higher reconstruction error because the model
hasn't learned their patterns.

Architecture:
    Input -> Encoder -> Latent Space -> Decoder -> Reconstruction
                            |
                    Bottleneck forces
                    learning of patterns

Advantages:
    - Learns complex patterns in data
    - Works with high-dimensional features
    - No distribution assumptions

Disadvantages:
    - Requires more training data
    - Needs hyperparameter tuning
    - Less interpretable than statistical methods

This implementation uses PyTorch for flexibility.
Falls back gracefully if PyTorch is not installed.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.anomaly.base import BaseDetector
from src.anomaly.statistical import AnomalyResult, AnomalySeverity, AnomalyType
from src.features.numeric import NumericFeatures
from src.features.temporal import TemporalFeatures

logger = logging.getLogger(__name__)

# Lazy import to avoid errors if not installed
_torch_available = None


def _check_torch():
    """Check if PyTorch is available."""
    global _torch_available
    if _torch_available is None:
        try:
            import torch  # noqa: F401

            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


@dataclass
class AutoencoderConfig:
    """Configuration for Autoencoder detector.

    Attributes:
        input_dim: Number of input features (set automatically from data).
        hidden_dims: List of hidden layer dimensions for encoder.
        latent_dim: Dimension of the latent space (bottleneck).
        learning_rate: Learning rate for training.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        anomaly_threshold: Reconstruction error threshold for anomalies.
        dropout: Dropout rate for regularization.
        use_embeddings: Whether embeddings are included in features.
        embedding_dim: Dimension of text embeddings (e.g., 384 for MiniLM).
        embedding_model: Name of the embedding model used.
    """

    input_dim: int = 0  # Set from data
    hidden_dims: list[int] | None = None  # Default: [64, 32] or [256, 128] with embeddings
    latent_dim: int = 8
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    anomaly_threshold: float = 0.5  # Percentile-based threshold
    dropout: float = 0.1
    use_embeddings: bool = False
    embedding_dim: int = 0
    embedding_model: str = ""

    def __post_init__(self):
        if self.hidden_dims is None:
            # Use larger hidden layers when embeddings are included
            if self.use_embeddings:
                self.hidden_dims = [256, 128]
            else:
                self.hidden_dims = [64, 32]


class AutoencoderModel:
    """PyTorch Autoencoder model for anomaly detection.

    This is a simple feedforward autoencoder with symmetric encoder/decoder.
    """

    def __init__(self, config: AutoencoderConfig):
        """Initialize the autoencoder model.

        Args:
            config: Configuration with architecture parameters.
        """
        if not _check_torch():
            raise ImportError(
                "PyTorch is required for AutoencoderModel. " "Install with: pip install torch"
            )

        import torch
        import torch.nn as nn

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build encoder layers
        encoder_layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, config.latent_dim))

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = config.latent_dim
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, config.input_dim))

        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, x):
        """Forward pass through encoder and decoder."""
        import torch

        x = torch.FloatTensor(x).to(self.device)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Get reconstruction error for input data.

        Args:
            x: Input data array.

        Returns:
            Reconstruction error per sample.
        """
        import torch

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            reconstructed = self.forward(x)
            # Mean squared error per sample
            errors = self.criterion(reconstructed, x_tensor).mean(dim=1)
            return errors.cpu().numpy()

    def fit(self, X: np.ndarray, verbose: bool = False, log_interval: int = 5):
        """Train the autoencoder on normal data.

        Args:
            X: Training data array (samples x features).
            verbose: Whether to print training progress.
            log_interval: Log progress every N epochs (default: 5). Set to 0 to disable.
        """
        import time

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.encoder.train()
        self.decoder.train()

        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(X))
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        n_batches = len(dataloader)

        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.config.learning_rate,
        )

        # Training loop with progress tracking
        start_time = time.perf_counter()
        epoch_times = []

        for epoch in range(self.config.epochs):
            epoch_start = time.perf_counter()
            total_loss = 0.0
            for batch in dataloader:
                x_batch = batch[0].to(self.device)

                # Forward pass
                reconstructed = self.decoder(self.encoder(x_batch))
                loss = self.criterion(reconstructed, x_batch).mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_batches
            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)

            # Log progress at specified intervals (always log, not just verbose)
            should_log = log_interval > 0 and (epoch + 1) % log_interval == 0
            # Also log first epoch to show training started
            is_first_epoch = epoch == 0 and log_interval > 0

            if should_log or is_first_epoch:
                elapsed = time.perf_counter() - start_time
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = self.config.epochs - (epoch + 1)
                eta_seconds = avg_epoch_time * remaining_epochs

                logger.info(
                    f"  Epoch {epoch + 1}/{self.config.epochs}: "
                    f"loss={avg_loss:.6f}, "
                    f"epoch_time={epoch_time:.1f}s, "
                    f"elapsed={elapsed:.1f}s, "
                    f"ETA={eta_seconds:.0f}s"
                )

            # Detailed verbose logging for debugging
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    "autoencoder_training",
                    extra={
                        "epoch": epoch + 1,
                        "total_epochs": self.config.epochs,
                        "avg_loss": avg_loss,
                    },
                )

        total_time = time.perf_counter() - start_time
        if log_interval > 0:
            logger.info(
                f"  Training complete: {self.config.epochs} epochs in {total_time:.1f}s "
                f"({total_time / self.config.epochs:.2f}s/epoch)"
            )

        self.encoder.eval()
        self.decoder.eval()


class AutoencoderDetector(BaseDetector):
    """Autoencoder-based anomaly detector.

    Learns the structure of normal data and flags records with
    high reconstruction error as anomalies.

    Usage:
        detector = AutoencoderDetector()
        detector.fit(training_features)
        result = detector.detect(numeric_features, temporal_features)
    """

    def __init__(self, config: AutoencoderConfig | None = None):
        """Initialize the autoencoder detector.

        Args:
            config: Configuration for the model. Uses defaults if None.
        """
        if not _check_torch():
            logger.warning(
                "torch_not_available: Autoencoder detector not available. "
                "Install with: pip install torch"
            )

        self.config = config or AutoencoderConfig()
        self.name = "autoencoder"
        self._model: AutoencoderModel | None = None
        self._is_fitted = False
        self._threshold: float = 0.0
        self._feature_names: list[str] = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def is_available(self) -> bool:
        """Check if autoencoder is available."""
        return _check_torch()

    def _prepare_features(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        embedding: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str], bool]:
        """Prepare feature vector for model input.

        Args:
            numeric_features: Numeric features from the record.
            temporal_features: Temporal features with rolling statistics.
            embedding: Optional text embedding vector (e.g., 384-dim from MiniLM).

        Returns:
            Tuple of (feature_vector, feature_names, is_valid).
        """
        feature_dict: dict[str, float | None] = {}

        # Numeric features
        feature_dict["price"] = numeric_features.price
        feature_dict["price_log"] = numeric_features.price_log
        feature_dict["price_ratio"] = numeric_features.price_ratio
        feature_dict["has_list_price"] = 1.0 if numeric_features.has_list_price else 0.0

        # Temporal features
        if temporal_features.has_sufficient_history:
            feature_dict["rolling_mean"] = temporal_features.rolling_mean
            feature_dict["rolling_std"] = temporal_features.rolling_std
            feature_dict["price_zscore"] = temporal_features.price_zscore
            feature_dict["price_change_pct"] = temporal_features.price_change_pct

            if temporal_features.rolling_mean and temporal_features.rolling_mean > 0:
                feature_dict["price_vs_mean_ratio"] = (
                    numeric_features.price / temporal_features.rolling_mean
                )
            else:
                feature_dict["price_vs_mean_ratio"] = 1.0
        else:
            for name in [
                "rolling_mean",
                "rolling_std",
                "price_zscore",
                "price_change_pct",
                "price_vs_mean_ratio",
            ]:
                feature_dict[name] = 0.0

        # Build numeric feature vector
        feature_names = list(feature_dict.keys())
        features = []
        is_valid = True

        for name in feature_names:
            value = feature_dict[name]
            if value is None or np.isnan(value) or np.isinf(value):
                features.append(0.0)
                is_valid = False
            else:
                features.append(float(value))

        numeric_array = np.array(features, dtype=np.float64)

        # Append embedding if provided
        if embedding is not None:
            # Add embedding feature names
            embedding_dim = len(embedding)
            for i in range(embedding_dim):
                feature_names.append(f"emb_{i}")

            # Check for NaN/Inf in embedding
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
                is_valid = False

            # Concatenate numeric features with embedding
            full_features = np.concatenate([numeric_array, embedding])
            return full_features, feature_names, is_valid

        return numeric_array, feature_names, is_valid

    def fit(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
        embeddings_list: list[np.ndarray] | None = None,
        embedding_model: str = "",
        verbose: bool = False,
    ) -> "AutoencoderDetector":
        """Fit the autoencoder on training data.

        Args:
            numeric_features_list: List of numeric features.
            temporal_features_list: List of temporal features.
            embeddings_list: Optional list of text embeddings (one per record).
            embedding_model: Name of the embedding model used (e.g., "all-MiniLM-L6-v2").
            verbose: Whether to print training progress.

        Returns:
            Self for method chaining.
        """
        if not self.is_available:
            raise RuntimeError("PyTorch not available")

        if len(numeric_features_list) != len(temporal_features_list):
            raise ValueError("Feature lists must have same length")

        use_embeddings = embeddings_list is not None and len(embeddings_list) > 0
        if use_embeddings and len(embeddings_list) != len(numeric_features_list):
            raise ValueError("Embeddings list must have same length as feature lists")

        # Update config for embeddings
        if use_embeddings:
            self.config.use_embeddings = True
            self.config.embedding_dim = len(embeddings_list[0])
            self.config.embedding_model = embedding_model
            # Use larger hidden layers for high-dimensional input
            if self.config.hidden_dims == [64, 32]:
                self.config.hidden_dims = [256, 128]

        # Prepare features
        features_list = []
        for i, (nf, tf) in enumerate(zip(numeric_features_list, temporal_features_list, strict=True)):
            embedding = embeddings_list[i] if use_embeddings else None
            features, names, is_valid = self._prepare_features(nf, tf, embedding)
            if is_valid:
                features_list.append(features)

        if len(features_list) < 50:
            raise ValueError(
                f"Need at least 50 valid samples for autoencoder, got {len(features_list)}"
            )

        self._feature_names = names
        X = np.vstack(features_list)

        # Normalize data
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0  # Avoid division by zero
        X_normalized = (X - self._mean) / self._std

        # Update config with input dimension
        self.config.input_dim = X.shape[1]

        logger.info(
            "autoencoder_fitting",
            extra={
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "use_embeddings": use_embeddings,
                "embedding_dim": self.config.embedding_dim if use_embeddings else 0,
                "epochs": self.config.epochs,
                "latent_dim": self.config.latent_dim,
            },
        )

        # Create and train model
        self._model = AutoencoderModel(self.config)
        self._model.fit(X_normalized, verbose=verbose)

        # Compute threshold based on training data reconstruction errors
        train_errors = self._model.get_reconstruction_error(X_normalized)
        self._threshold = np.percentile(train_errors, 95)  # 95th percentile

        self._is_fitted = True

        logger.info(
            "autoencoder_fitted",
            extra={
                "n_samples": X.shape[0],
                "threshold": self._threshold,
                "mean_error": train_errors.mean(),
                "max_error": train_errors.max(),
                "use_embeddings": use_embeddings,
            },
        )

        return self

    def detect(
        self,
        numeric_features: NumericFeatures,
        temporal_features: TemporalFeatures,
        embedding: np.ndarray | None = None,
    ) -> AnomalyResult:
        """Detect anomalies using the trained autoencoder.

        Args:
            numeric_features: Numeric features from the record.
            temporal_features: Temporal features.
            embedding: Optional text embedding (required if model was trained with embeddings).

        Returns:
            AnomalyResult with detection results.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before detection")

        # Validate embedding requirement
        if self.config.use_embeddings and embedding is None:
            raise ValueError("Model was trained with embeddings but none provided")

        # Prepare features
        features, feature_names, is_valid = self._prepare_features(
            numeric_features, temporal_features, embedding
        )

        # Validate schema
        if self._feature_names:
            from src.anomaly.ml import validate_feature_schema

            validate_feature_schema(feature_names, self._feature_names, "Autoencoder")

        features = features.reshape(1, -1)

        # Normalize
        features_normalized = (features - self._mean) / self._std

        # Get reconstruction error
        error = self._model.get_reconstruction_error(features_normalized)[0]

        # Normalize to 0-1 score using base class method
        anomaly_score = self.normalize_score(error, self._threshold)

        details: dict[str, Any] = {
            "reconstruction_error": float(error),
            "threshold": self._threshold,
            "feature_valid": is_valid,
            "use_embeddings": self.config.use_embeddings,
        }

        # Determine if anomaly
        is_anomaly = error > self._threshold
        anomaly_types: list[AnomalyType] = []
        severity = None

        if is_anomaly:
            anomaly_types.append(AnomalyType.PRICE_ZSCORE)  # Generic type

            if error > self._threshold * 3:
                severity = AnomalySeverity.CRITICAL
            elif error > self._threshold * 2:
                severity = AnomalySeverity.HIGH
            elif error > self._threshold * 1.5:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW

        if is_anomaly:
            logger.info(
                "anomaly_detected",
                extra={
                    "detector": self.name,
                    "competitor_product_id": numeric_features.competitor_product_id,
                    "competitor": numeric_features.competitor,
                    "price": numeric_features.price,
                    "anomaly_score": anomaly_score,
                    "reconstruction_error": error,
                    "severity": severity.value if severity else None,
                },
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_types=anomaly_types,
            severity=severity,
            details=details,
            detector=self.name,
            competitor_product_id=numeric_features.competitor_product_id,
            competitor=numeric_features.competitor,
        )

    def detect_batch(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
        embeddings_list: list[np.ndarray] | None = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies for a batch of records.

        Args:
            numeric_features_list: List of numeric features.
            temporal_features_list: List of temporal features.
            embeddings_list: Optional list of text embeddings (required if model was trained with embeddings).

        Returns:
            List of AnomalyResult.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before detection")

        # Validate embedding requirement
        if self.config.use_embeddings:
            if embeddings_list is None or len(embeddings_list) != len(numeric_features_list):
                raise ValueError(
                    "Model was trained with embeddings - must provide embeddings_list "
                    "with same length as feature lists"
                )

        # Prepare all features
        features_list = []
        valid_flags = []
        for i, (nf, tf) in enumerate(zip(numeric_features_list, temporal_features_list, strict=True)):
            embedding = embeddings_list[i] if embeddings_list else None
            features, feature_names, is_valid = self._prepare_features(nf, tf, embedding)
            features_list.append(features)
            valid_flags.append(is_valid)

            # Validate schema on first feature (all should have same names)
            if i == 0 and self._feature_names:
                from src.anomaly.ml import validate_feature_schema

                validate_feature_schema(feature_names, self._feature_names, "Autoencoder")

        X = np.vstack(features_list)
        X_normalized = (X - self._mean) / self._std

        # Get all errors at once
        errors = self._model.get_reconstruction_error(X_normalized)

        # Build results
        results = []
        for i, (nf, error, is_valid) in enumerate(zip(numeric_features_list, errors, valid_flags)):
            anomaly_score = self.normalize_score(error, self._threshold)
            is_anomaly = error > self._threshold

            anomaly_types: list[AnomalyType] = []
            severity = None

            if is_anomaly:
                anomaly_types.append(AnomalyType.PRICE_ZSCORE)
                if error > self._threshold * 3:
                    severity = AnomalySeverity.CRITICAL
                elif error > self._threshold * 2:
                    severity = AnomalySeverity.HIGH
                elif error > self._threshold * 1.5:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW

            results.append(
                AnomalyResult(
                    is_anomaly=is_anomaly,
                    anomaly_score=float(anomaly_score),
                    anomaly_types=anomaly_types,
                    severity=severity,
                    details={
                        "reconstruction_error": float(error),
                        "threshold": self._threshold,
                        "use_embeddings": self.config.use_embeddings,
                        "is_valid_input": is_valid,
                    },
                    detector=self.name,
                    competitor_product_id=nf.competitor_product_id,
                    competitor=nf.competitor,
                )
            )

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "autoencoder_batch_detection",
            extra={
                "total_records": len(results),
                "anomalies_detected": anomaly_count,
                "anomaly_rate": anomaly_count / len(results) if results else 0,
                "use_embeddings": self.config.use_embeddings,
            },
        )

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the trained model."""
        if not self._is_fitted:
            return {"is_fitted": False, "is_available": self.is_available}

        return {
            "is_fitted": True,
            "is_available": self.is_available,
            "input_dim": self.config.input_dim,
            "hidden_dims": self.config.hidden_dims,
            "latent_dim": self.config.latent_dim,
            "threshold": self._threshold,
            "feature_names": self._feature_names,
            "use_embeddings": self.config.use_embeddings,
            "embedding_dim": self.config.embedding_dim,
            "embedding_model": self.config.embedding_model,
        }

    def calibrate_threshold(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
        embeddings_list: list[np.ndarray] | None = None,
        percentile: float = 95.0,
    ) -> float:
        """Calibrate anomaly threshold on new data without retraining.

        This method adjusts the decision boundary based on the reconstruction
        error distribution of new data. Use this when applying a pre-trained
        model to a new data source that may have different characteristics.

        Three-way comparison for thesis:
        1. Uncalibrated: Use original threshold from training
        2. Calibrated: Adjust threshold via this method (quick)
        3. Retrained: Full retraining on combined data (expensive)

        Args:
            numeric_features_list: List of numeric features from new data.
            temporal_features_list: List of temporal features from new data.
            embeddings_list: Optional text embeddings (required if model uses them).
            percentile: Percentile of reconstruction errors to use as threshold.
                Default 95.0 means 5% of new data will be flagged as anomalies.

        Returns:
            The new threshold value.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If embeddings are required but not provided.

        Example:
            # Load pre-trained model (e.g., trained on internal data)
            detector = AutoencoderDetector.load("internal_model.pt")
            original_threshold = detector._threshold

            # Calibrate on new competitor data
            new_threshold = detector.calibrate_threshold(
                competitor_numeric_features,
                competitor_temporal_features,
                percentile=95.0,
            )

            # Compare detection with original vs calibrated threshold
            print(f"Original: {original_threshold:.4f}, Calibrated: {new_threshold:.4f}")
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calibration")

        if self.config.use_embeddings:
            if embeddings_list is None or len(embeddings_list) != len(numeric_features_list):
                raise ValueError(
                    "Model was trained with embeddings - must provide embeddings_list "
                    "with same length as feature lists"
                )

        # Store original threshold for logging
        original_threshold = self._threshold

        # Prepare features
        features_list = []
        for i, (nf, tf) in enumerate(zip(numeric_features_list, temporal_features_list, strict=True)):
            embedding = embeddings_list[i] if embeddings_list else None
            features, _, is_valid = self._prepare_features(nf, tf, embedding)
            if is_valid:
                features_list.append(features)

        if len(features_list) < 10:
            logger.warning(
                f"calibrate_threshold: only {len(features_list)} valid samples, "
                "threshold may be unreliable"
            )

        if not features_list:
            raise ValueError("No valid samples for calibration")

        # Compute reconstruction errors on new data
        X = np.vstack(features_list)
        X_normalized = (X - self._mean) / self._std
        errors = self._model.get_reconstruction_error(X_normalized)

        # Set new threshold at specified percentile
        self._threshold = float(np.percentile(errors, percentile))

        logger.info(
            "autoencoder_threshold_calibrated",
            extra={
                "original_threshold": original_threshold,
                "new_threshold": self._threshold,
                "percentile": percentile,
                "n_samples": len(features_list),
                "mean_error": float(errors.mean()),
                "max_error": float(errors.max()),
                "use_embeddings": self.config.use_embeddings,
            },
        )

        return self._threshold

    def get_reconstruction_errors(
        self,
        numeric_features_list: list[NumericFeatures],
        temporal_features_list: list[TemporalFeatures],
        embeddings_list: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Get reconstruction errors for a batch of records.

        Useful for analyzing error distributions before calibration.

        Args:
            numeric_features_list: List of numeric features.
            temporal_features_list: List of temporal features.
            embeddings_list: Optional text embeddings.

        Returns:
            Array of reconstruction errors (one per sample).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before computing errors")

        if self.config.use_embeddings:
            if embeddings_list is None or len(embeddings_list) != len(numeric_features_list):
                raise ValueError(
                    "Model was trained with embeddings - must provide embeddings_list"
                )

        features_list = []
        for i, (nf, tf) in enumerate(zip(numeric_features_list, temporal_features_list, strict=True)):
            embedding = embeddings_list[i] if embeddings_list else None
            features, _, _ = self._prepare_features(nf, tf, embedding)
            features_list.append(features)

        X = np.vstack(features_list)
        X_normalized = (X - self._mean) / self._std
        return self._model.get_reconstruction_error(X_normalized)

    def save(self, path: str) -> str:
        """Save trained model to a local file.

        Args:
            path: Local file path (should end with .pt).

        Returns:
            Path where model was saved.

        Raises:
            ValueError: If model is not fitted.
            RuntimeError: If PyTorch is not available.
        """
        if not self.is_available:
            raise RuntimeError("PyTorch not available")

        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")

        import torch

        save_data = {
            "encoder_state_dict": self._model.encoder.state_dict(),
            "decoder_state_dict": self._model.decoder.state_dict(),
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dims": self.config.hidden_dims,
                "latent_dim": self.config.latent_dim,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "anomaly_threshold": self.config.anomaly_threshold,
                "dropout": self.config.dropout,
                "use_embeddings": self.config.use_embeddings,
                "embedding_dim": self.config.embedding_dim,
                "embedding_model": self.config.embedding_model,
            },
            "threshold": self._threshold,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
            "feature_names": self._feature_names,
        }

        torch.save(save_data, path)
        logger.info(
            "autoencoder_saved_local",
            extra={
                "path": path,
                "input_dim": self.config.input_dim,
                "threshold": self._threshold,
                "use_embeddings": self.config.use_embeddings,
                "embedding_model": self.config.embedding_model,
            },
        )
        return path

    @classmethod
    def load(cls, path: str) -> "AutoencoderDetector":
        """Load a trained model from a local file.

        Args:
            path: Local file path to the saved model.

        Returns:
            Loaded AutoencoderDetector ready for inference.

        Raises:
            RuntimeError: If PyTorch is not available.
        """
        if not _check_torch():
            raise RuntimeError("PyTorch not available")

        import torch

        saved_data = torch.load(path, map_location="cpu", weights_only=False)

        # Reconstruct config (with backward compatibility for models without embeddings)
        config_dict = saved_data["config"]
        config = AutoencoderConfig(
            input_dim=config_dict["input_dim"],
            hidden_dims=config_dict["hidden_dims"],
            latent_dim=config_dict["latent_dim"],
            learning_rate=config_dict["learning_rate"],
            epochs=config_dict["epochs"],
            batch_size=config_dict["batch_size"],
            anomaly_threshold=config_dict["anomaly_threshold"],
            dropout=config_dict["dropout"],
            use_embeddings=config_dict.get("use_embeddings", False),
            embedding_dim=config_dict.get("embedding_dim", 0),
            embedding_model=config_dict.get("embedding_model", ""),
        )

        # Create detector and restore state
        detector = cls(config)
        detector._model = AutoencoderModel(config)
        detector._model.encoder.load_state_dict(saved_data["encoder_state_dict"])
        detector._model.decoder.load_state_dict(saved_data["decoder_state_dict"])
        detector._model.encoder.eval()
        detector._model.decoder.eval()

        detector._threshold = saved_data["threshold"]
        if saved_data["mean"] is not None:
            detector._mean = np.array(saved_data["mean"])
        if saved_data["std"] is not None:
            detector._std = np.array(saved_data["std"])
        detector._feature_names = saved_data["feature_names"]
        detector._is_fitted = True

        logger.info(
            "autoencoder_loaded_local",
            extra={
                "path": path,
                "input_dim": config.input_dim,
                "threshold": detector._threshold,
                "use_embeddings": config.use_embeddings,
                "embedding_model": config.embedding_model,
            },
        )
        return detector
