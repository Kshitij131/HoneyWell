"""Core neural network models, Policy Engine, Platt Scaler, and Hybrid Risk Scorer.

Contains:
- SharedAutoencoder: Deep reconstruction neural network for unsupervised anomaly detection.
- LSTMSequenceClassifier: Recurrent neural network with Focal Loss for temporal sequence classification.
- PolicyEngine: Domain-specific security rule scoring engine.
- HybridRiskScorer: Weighted risk score aggregator (R = w_ae*AE + w_lstm*LSTM + w_policy*Policy).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ENTITY_TYPES,
    AutoencoderConfig,
    FocalLossConfig,
    AttackClassifierConfig,
    ModelConfig,
    PolicyEngineConfig,
    PolicyRule,
    ProjectConfig,
    RiskScorerConfig,
    TrainingConfig,
    get_project_config,
)

logger = logging.getLogger(__name__)

def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("PyTorch seeds set to %d", seed)

def _get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if device_str == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)

class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()

    @classmethod
    def from_config(cls, cfg: FocalLossConfig) -> "FocalLoss":
        return cls(alpha=cfg.alpha, gamma=cfg.gamma)

class MultiClassFocalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.class_weights, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

    @staticmethod
    def compute_class_weights(
        labels: np.ndarray,
        num_classes: int,
        max_ratio: float = 10.0,
    ) -> torch.Tensor:
        counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        inv_sqrt = 1.0 / np.sqrt(counts)
        inv_sqrt /= inv_sqrt.min()
        inv_sqrt = np.clip(inv_sqrt, 1.0, max_ratio)
        return torch.tensor(inv_sqrt, dtype=torch.float32)

    @classmethod
    def from_config(cls, cfg: AttackClassifierConfig) -> "MultiClassFocalLoss":
        return cls(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

class SharedAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        cfg: AutoencoderConfig,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.cfg: AutoencoderConfig = cfg

        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in cfg.encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(self._get_activation(cfg.activation))
            encoder_layers.append(nn.Dropout(cfg.dropout_rate))
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        prev_dim = cfg.latent_dim
        for dim in cfg.decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(self._get_activation(cfg.activation))
            decoder_layers.append(nn.Dropout(cfg.dropout_rate))
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.thresholds: Dict[str, float] = {}

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
        }
        if name not in activations:
            raise ValueError(
                f"Unsupported activation: {name}. Choose from {list(activations.keys())}"
            )
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed = self.forward(x)
        mse = ((x - reconstructed) ** 2).mean(dim=1)
        return mse

    def compute_adaptive_thresholds(
        self,
        X: np.ndarray,
        entity_types: np.ndarray,
        percentile: float,
        device: torch.device,
    ) -> Dict[str, float]:
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            errors = self.compute_reconstruction_error(X_tensor).cpu().numpy()

        thresholds: Dict[str, float] = {}
        for etype in ENTITY_TYPES:
            mask = entity_types == etype
            if mask.sum() > 0:
                threshold = float(np.percentile(errors[mask], percentile))
                thresholds[etype] = threshold
                logger.info(
                    "Threshold for %s: %.6f (p%.0f, n=%d)",
                    etype,
                    threshold,
                    percentile,
                    mask.sum(),
                )
            else:
                thresholds[etype] = float(np.percentile(errors, percentile))
                logger.warning(
                    "No samples for %s; using global threshold %.6f",
                    etype,
                    thresholds[etype],
                )

        self.thresholds = thresholds
        return thresholds

    @classmethod
    def from_config(cls, input_dim: int, cfg: AutoencoderConfig) -> "SharedAutoencoder":
        return cls(input_dim=input_dim, cfg=cfg)

class LSTMSequenceClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        cfg: LSTMConfig,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.cfg: LSTMConfig = cfg

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout_rate if cfg.num_layers > 1 else 0.0,
        )

        lstm_output_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)

        fc_layers: List[nn.Module] = []
        prev_dim = lstm_output_dim
        for dim in cfg.fc_dims:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.BatchNorm1d(dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(cfg.dropout_rate))
            prev_dim = dim
        self.fc_layers = nn.Sequential(*fc_layers)

        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]

        fc_out = self.fc_layers(last_hidden)

        logits = self.output_layer(fc_out).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def extract_hidden(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]

    @classmethod
    def from_config(cls, input_dim: int, cfg: LSTMConfig) -> "LSTMSequenceClassifier":
        return cls(input_dim=input_dim, cfg=cfg)

class AttackTypeClassifier(nn.Module):

    def __init__(self, input_dim: int, cfg: AttackClassifierConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.cfg = cfg
        
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in cfg.fc_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout_rate))
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, cfg.num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @classmethod
    def from_config(cls, input_dim: int, cfg: AttackClassifierConfig) -> "AttackTypeClassifier":
        return cls(input_dim=input_dim, cfg=cfg)

class PolicyEngine:

    _OPERATORS = {
        "gt": lambda x, t: x > t,
        "lt": lambda x, t: x < t,
        "ge": lambda x, t: x >= t,
        "le": lambda x, t: x <= t,
        "eq": lambda x, t: np.isclose(x, t),
        "ne": lambda x, t: ~np.isclose(x, t),
    }

    def __init__(self, cfg: PolicyEngineConfig) -> None:
        self.rules: Tuple[PolicyRule, ...] = cfg.rules
        self.max_score: float = cfg.max_score
        logger.info(
            "PolicyEngine initialised with %d rules, max_score=%.2f",
            len(self.rules),
            self.max_score,
        )

    def evaluate(
        self,
        features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        n_samples = next(iter(features.values())).shape[0]
        raw_scores = np.zeros(n_samples, dtype=np.float64)

        for rule in self.rules:
            if rule.feature not in features:
                logger.warning(
                    "Feature '%s' not found for rule '%s'. Skipping.",
                    rule.feature,
                    rule.name,
                )
                continue

            feature_values = features[rule.feature]
            operator_fn = self._OPERATORS.get(rule.operator)
            if operator_fn is None:
                logger.warning(
                    "Unknown operator '%s' in rule '%s'. Skipping.",
                    rule.operator,
                    rule.name,
                )
                continue

            mask = operator_fn(feature_values, rule.threshold)
            raw_scores[mask] += rule.score

        normalised = np.clip(raw_scores / self.max_score, 0.0, 1.0)
        return normalised

    def evaluate_single(self, features: Dict[str, float]) -> float:
        raw_score = 0.0
        for rule in self.rules:
            if rule.feature not in features:
                continue
            val = features[rule.feature]
            operator_fn = self._OPERATORS.get(rule.operator)
            if operator_fn is None:
                continue
            if operator_fn(np.float64(val), rule.threshold):
                raw_score += rule.score
        return min(raw_score / self.max_score, 1.0)

    @classmethod
    def from_config(cls, cfg: PolicyEngineConfig) -> "PolicyEngine":
        return cls(cfg=cfg)

class RiskScorer:

    def __init__(self, cfg: RiskScorerConfig) -> None:
        self.w_ae: float = cfg.w_ae
        self.w_lstm: float = cfg.w_lstm
        self.w_policy: float = cfg.w_policy

        total = self.w_ae + self.w_lstm + self.w_policy
        if not np.isclose(total, 1.0):
            logger.warning(
                "Risk scorer weights sum to %.4f, not 1.0. "
                "Scores will still be computed but may exceed [0, 1].",
                total,
            )

    def compute(
        self,
        ae_errors: np.ndarray,
        lstm_probs: np.ndarray,
        policy_scores: np.ndarray,
        ae_min: float = 0.0,
        ae_max: float = 1.0,
    ) -> np.ndarray:
        if ae_max > ae_min:
            normalised_ae = (ae_errors - ae_min) / (ae_max - ae_min)
        else:
            normalised_ae = np.zeros_like(ae_errors)
        normalised_ae = np.clip(normalised_ae, 0.0, 1.0)

        normalised_lstm = np.clip(lstm_probs, 0.0, 1.0)

        normalised_policy = np.clip(policy_scores, 0.0, 1.0)

        risk = (
            self.w_ae * normalised_ae
            + self.w_lstm * normalised_lstm
            + self.w_policy * normalised_policy
        )

        return np.clip(risk, 0.0, 1.0)

    def compute_single(
        self,
        ae_error: float,
        lstm_prob: float,
        policy_score: float,
        ae_min: float,
        ae_max: float,
    ) -> float:
        if ae_max > ae_min:
            norm_ae = (ae_error - ae_min) / (ae_max - ae_min)
        else:
            norm_ae = 0.0
        norm_ae = max(0.0, min(1.0, norm_ae))
        norm_lstm = max(0.0, min(1.0, lstm_prob))
        norm_policy = max(0.0, min(1.0, policy_score))

        risk = self.w_ae * norm_ae + self.w_lstm * norm_lstm + self.w_policy * norm_policy
        return max(0.0, min(1.0, risk))

    def update_weights(self, w_ae: float, w_lstm: float, w_policy: float) -> None:
        self.w_ae = w_ae
        self.w_lstm = w_lstm
        self.w_policy = w_policy

    @classmethod
    def from_config(cls, cfg: RiskScorerConfig) -> "RiskScorer":
        return cls(cfg=cfg)

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = len(X)
    if n_samples < sequence_length:
        logger.warning(
            "Not enough samples (%d) for sequence_length (%d). "
            "Returning empty arrays.",
            n_samples,
            sequence_length,
        )
        return (
            np.empty((0, sequence_length, X.shape[1]), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    n_sequences = n_samples - sequence_length + 1
    X_seq = np.zeros(
        (n_sequences, sequence_length, X.shape[1]), dtype=np.float64
    )
    y_seq = np.zeros(n_sequences, dtype=np.float64)

    for i in range(n_sequences):
        X_seq[i] = X[i : i + sequence_length]
        y_seq[i] = y[i + sequence_length - 1]

    logger.info(
        "Created %d sequences of length %d from %d samples.",
        n_sequences,
        sequence_length,
        n_samples,
    )
    return X_seq, y_seq

def train_autoencoder(
    model: SharedAutoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    training_cfg: TrainingConfig,
) -> Dict[str, List[float]]:
    device = _get_device(training_cfg.device)
    model = model.to(device)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.learning_rate_ae,
        weight_decay=training_cfg.weight_decay,
    )
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {"train_losses": [], "val_losses": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    logger.info(
        "Training SharedAutoencoder — epochs: %d, batch_size: %d, device: %s",
        training_cfg.num_epochs_ae,
        training_cfg.batch_size,
        device,
    )

    for epoch in range(training_cfg.num_epochs_ae):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch_x, batch_target in train_loader:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_cfg.gradient_clip_max_norm
            )
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_target in val_loader:
                batch_x = batch_x.to(device)
                batch_target = batch_target.to(device)
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_target)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        history["train_losses"].append(avg_train_loss)
        history["val_losses"].append(avg_val_loss)

        logger.info(
            "Epoch %d/%d — train_loss: %.6f, val_loss: %.6f",
            epoch + 1,
            training_cfg.num_epochs_ae,
            avg_train_loss,
            avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= training_cfg.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1,
                    training_cfg.early_stopping_patience,
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model = model.to(device)
        logger.info("Restored best model weights (val_loss=%.6f)", best_val_loss)

    return history

def train_lstm(
    model: LSTMSequenceClassifier,
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    focal_loss: FocalLoss,
    training_cfg: TrainingConfig,
) -> Dict[str, List[float]]:
    device = _get_device(training_cfg.device)
    model = model.to(device)
    focal_loss = focal_loss.to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_seq, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.learning_rate_lstm,
        weight_decay=training_cfg.weight_decay,
    )

    history: Dict[str, List[float]] = {"train_losses": [], "val_losses": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    logger.info(
        "Training LSTMSequenceClassifier — epochs: %d, batch_size: %d, device: %s",
        training_cfg.num_epochs_lstm,
        training_cfg.batch_size,
        device,
    )

    for epoch in range(training_cfg.num_epochs_lstm):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = focal_loss(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_cfg.gradient_clip_max_norm
            )
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = focal_loss(logits, batch_y)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        history["train_losses"].append(avg_train_loss)
        history["val_losses"].append(avg_val_loss)

        logger.info(
            "Epoch %d/%d — train_loss: %.6f, val_loss: %.6f",
            epoch + 1,
            training_cfg.num_epochs_lstm,
            avg_train_loss,
            avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= training_cfg.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1,
                    training_cfg.early_stopping_patience,
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model = model.to(device)
        logger.info("Restored best model weights (val_loss=%.6f)", best_val_loss)

    return history

def train_attack_classifier(
    model: AttackTypeClassifier,
    X_train_hidden: np.ndarray,
    y_train_attack: np.ndarray,
    X_val_hidden: np.ndarray,
    y_val_attack: np.ndarray,
    focal_loss: MultiClassFocalLoss,
    training_cfg: TrainingConfig,
) -> Dict[str, List[float]]:
    device = _get_device(training_cfg.device)
    model = model.to(device)
    focal_loss = focal_loss.to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train_hidden, dtype=torch.float32),
        torch.tensor(y_train_attack, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_hidden, dtype=torch.float32),
        torch.tensor(y_val_attack, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model.cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )

    history: Dict[str, List[float]] = {"train_losses": [], "val_losses": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    logger.info("Training AttackTypeClassifier — epochs: %d, device: %s", model.cfg.num_epochs, device)

    for epoch in range(model.cfg.num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = focal_loss(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = focal_loss(logits, batch_y)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        history["train_losses"].append(avg_train_loss)
        history["val_losses"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= training_cfg.early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model = model.to(device)

    return history

def infer_autoencoder(
    model: SharedAutoencoder,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    errors: List[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            batch_errors = model.compute_reconstruction_error(batch)
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors, axis=0)

def infer_lstm(
    model: LSTMSequenceClassifier,
    X_seq: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)

    probs: List[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            batch_probs = model.predict_proba(batch)
            probs.append(batch_probs.cpu().numpy())

    return np.concatenate(probs, axis=0)

def explain_autoencoder(
    model: SharedAutoencoder,
    X: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    n_steps: int = 50,
) -> Dict[str, np.ndarray]:
    model.eval()
    model = model.to(device)

    def ae_forward_for_ig(x: torch.Tensor) -> torch.Tensor:
        reconstructed = model(x)
        error = ((x - reconstructed) ** 2).mean(dim=1)
        return error

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    X_tensor.requires_grad_(True)

    baseline = torch.zeros_like(X_tensor, device=device)

    ig = IntegratedGradients(ae_forward_for_ig)

    batch_size = 256
    all_attributions: List[np.ndarray] = []

    for i in range(0, len(X_tensor), batch_size):
        batch_input = X_tensor[i : i + batch_size]
        batch_baseline = baseline[i : i + batch_size]

        attrs = ig.attribute(
            batch_input,
            baselines=batch_baseline,
            n_steps=n_steps,
            return_convergence_delta=False,
        )
        all_attributions.append(attrs.detach().cpu().numpy())

    attributions = np.concatenate(all_attributions, axis=0)
    mean_abs_attr = np.mean(np.abs(attributions), axis=0)

    sorted_indices = np.argsort(mean_abs_attr)[::-1]
    logger.info("Top 5 features by mean |IG attribution|:")
    for rank, idx in enumerate(sorted_indices[:5]):
        logger.info(
            "  %d. %s: %.6f", rank + 1, feature_names[idx], mean_abs_attr[idx]
        )

    return {
        "attributions": attributions,
        "feature_names": feature_names,
        "mean_attributions": mean_abs_attr,
    }

def save_model(
    model: nn.Module,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: Dict[str, Any] = {"state_dict": model.state_dict()}
    if metadata is not None:
        save_dict["metadata"] = metadata

    torch.save(save_dict, path)
    logger.info("Saved model to %s", path)

def load_model(
    model: nn.Module,
    path: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    path = Path(path)
    map_location = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model.to(device)

    logger.info("Loaded model from %s", path)
    return checkpoint.get("metadata", {})

def save_thresholds(thresholds: Dict[str, float], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Saved thresholds to %s", path)

def load_thresholds(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    logger.info("Loaded thresholds from %s", path)
    return thresholds

def run_hybrid_inference(
    autoencoder: SharedAutoencoder,
    lstm_classifier: LSTMSequenceClassifier,
    policy_engine: PolicyEngine,
    risk_scorer: RiskScorer,
    X: np.ndarray,
    feature_names: List[str],
    sequence_length: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    ae_errors = infer_autoencoder(autoencoder, X, device)

    dummy_y = np.zeros(len(X), dtype=np.float64)
    X_seq, _ = create_sequences(X, dummy_y, sequence_length)
    lstm_probs = infer_lstm(lstm_classifier, X_seq, device)

    offset = sequence_length - 1
    aligned_X = X[offset:]
    feature_dict: Dict[str, np.ndarray] = {}
    for i, name in enumerate(feature_names):
        feature_dict[name] = aligned_X[:, i]
    policy_scores = policy_engine.evaluate(feature_dict)

    ae_errors_aligned = ae_errors[offset:]

    min_len = min(len(ae_errors_aligned), len(lstm_probs), len(policy_scores))
    ae_errors_aligned = ae_errors_aligned[:min_len]
    lstm_probs = lstm_probs[:min_len]
    policy_scores = policy_scores[:min_len]

    risk_scores = risk_scorer.compute(ae_errors_aligned, lstm_probs, policy_scores)

    logger.info(
        "Hybrid inference complete — %d samples scored. "
        "Risk score stats: mean=%.4f, std=%.4f, max=%.4f",
        min_len,
        risk_scores.mean(),
        risk_scores.std(),
        risk_scores.max(),
    )

    return {
        "ae_errors": ae_errors_aligned,
        "lstm_probs": lstm_probs,
        "policy_scores": policy_scores,
        "risk_scores": risk_scores,
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger.info("models.py loaded. Use the public API to train and infer.")
