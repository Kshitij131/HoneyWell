"""Command-line orchestrator for executing individual or end-to-end pipeline stages.

Supported stages: generate, features, train, calibrate, infer, all.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import torch

from config import (
    ENTITY_TYPES,
    ATTACK_TYPES,
    NUM_ENTITY_TYPES,
    ATTACK_TYPE_LABELS,
    ProjectConfig,
    get_project_config,
)
from data_generator import generate_dataset
from feature_pipeline import run_feature_pipeline
from models import (
    FocalLoss,
    LSTMSequenceClassifier,
    PolicyEngine,
    RiskScorer,
    SharedAutoencoder,
    create_sequences,
    infer_autoencoder,
    save_model,
    save_thresholds,
    set_torch_seed,
    train_attack_classifier,
    train_autoencoder,
    train_lstm,
    AttackTypeClassifier,
    MultiClassFocalLoss,
)
from models import _get_device
from inference_pipeline import InferencePipeline
from evaluation import generate_report, find_optimal_threshold
from signature_rules import tune_signatures

logger = logging.getLogger(__name__)

def run_stage_generate(config: ProjectConfig) -> None:
    logger.info("=" * 60)
    logger.info("STAGE 1: Synthetic Data Generation")
    logger.info("=" * 60)
    start = time.perf_counter()
    generate_dataset(config)
    elapsed = time.perf_counter() - start
    logger.info("Data generation completed in %.2f seconds.", elapsed)

def run_stage_features(config: ProjectConfig) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("STAGE 2: Feature Engineering")
    logger.info("=" * 60)
    start = time.perf_counter()
    result = run_feature_pipeline(config)
    elapsed = time.perf_counter() - start
    logger.info("Feature engineering completed in %.2f seconds.", elapsed)
    return result

def run_stage_train(config: ProjectConfig) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("STAGE 3: Model Training")
    logger.info("=" * 60)
    start = time.perf_counter()

    paths = config.paths
    proc_dir = paths.processed_data_dir
    models_dir = paths.models_dir
    model_cfg = config.model
    training_cfg = config.training
    feature_cfg = config.features

    set_torch_seed(training_cfg.random_seed)
    device = _get_device(training_cfg.device)

    X_train = np.load(proc_dir / feature_cfg.x_train_file)
    X_val = np.load(proc_dir / feature_cfg.x_val_file)
    y_train = np.load(proc_dir / feature_cfg.y_train_file)
    y_val = np.load(proc_dir / feature_cfg.y_val_file)

    with open(proc_dir / feature_cfg.feature_metadata_file, "r") as f:
        metadata = json.load(f)
    num_features = metadata["num_features"]

    logger.info(
        "Loaded training data: X_train=%s, X_val=%s",
        X_train.shape, X_val.shape,
    )

    logger.info("Training SharedAutoencoder...")
    autoencoder = SharedAutoencoder.from_config(
        input_dim=num_features, cfg=model_cfg.autoencoder
    )
    ae_history = train_autoencoder(
        autoencoder, X_train, X_val, training_cfg
    )
    save_model(
        autoencoder,
        models_dir / "autoencoder.pt",
        metadata={"input_dim": num_features, "config": "autoencoder"},
    )

    df_logs = pd.read_csv(paths.raw_logs_file, parse_dates=["timestamp"])
    df_logs = df_logs.sort_values("timestamp").reset_index(drop=True)
    n_total = len(df_logs)
    train_end = int(n_total * feature_cfg.train_ratio)
    train_entity_types = df_logs.iloc[:train_end]["entity_type"].values

    thresholds = autoencoder.compute_adaptive_thresholds(
        X_train, train_entity_types, model_cfg.autoencoder.threshold_percentile, device
    )
    save_thresholds(thresholds, models_dir / "thresholds.json")
    
    logger.info("Computing training AE errors for drift reference...")
    train_ae_errors = infer_autoencoder(autoencoder, X_train, device)
    np.savez(
        models_dir / "ae_reference_stats.npz", 
        errors=train_ae_errors, 
        entity_types=train_entity_types
    )
    logger.info("Saved AE reference stats to ae_reference_stats.npz")

    logger.info("Training LSTMSequenceClassifier...")
    X_train_seq, y_train_seq = create_sequences(
        X_train, y_train, model_cfg.lstm.sequence_length
    )
    X_val_seq, y_val_seq = create_sequences(
        X_val, y_val, model_cfg.lstm.sequence_length
    )

    lstm = LSTMSequenceClassifier.from_config(
        input_dim=num_features, cfg=model_cfg.lstm
    )
    focal_loss = FocalLoss.from_config(model_cfg.focal_loss)
    lstm_history = train_lstm(
        lstm, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
        focal_loss, training_cfg,
    )
    save_model(
        lstm,
        models_dir / "lstm_classifier.pt",
        metadata={"input_dim": num_features, "config": "lstm"},
    )

    logger.info("Fitting Platt Scaling calibrator on validation logits...")
    lstm.eval()
    val_tensor_seq = torch.tensor(X_val_seq, dtype=torch.float32, device=device)
    with torch.no_grad():
        val_logits = lstm(val_tensor_seq).cpu().numpy()
        
    platt_scaler = LogisticRegression(solver="lbfgs")
    platt_scaler.fit(val_logits.reshape(-1, 1), y_val_seq)
    with open(paths.platt_scaler_file, "wb") as f:
        pickle.dump(platt_scaler, f)
        
    val_probs = platt_scaler.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    
    logger.info("Extracting frozen LSTM features for Attack Type Classifier...")
    df_truth = pd.read_csv(paths.ground_truth_file, parse_dates=["timestamp"])
    df_truth = df_truth.sort_values("timestamp").reset_index(drop=True)
    val_end = train_end + int(n_total * feature_cfg.val_ratio)
    
    train_attack_types = df_truth.iloc[:train_end]["attack_type"].values
    val_attack_types = df_truth.iloc[train_end:val_end]["attack_type"].values
    
    train_attack_labels = np.array([ATTACK_TYPE_LABELS.get(at, 0) for at in train_attack_types])
    val_attack_labels = np.array([ATTACK_TYPE_LABELS.get(at, 0) for at in val_attack_types])
    
    offset = model_cfg.lstm.sequence_length - 1
    train_attack_labels_seq = train_attack_labels[offset:]
    val_attack_labels_seq = val_attack_labels[offset:]
    
    lstm.eval()
    with torch.no_grad():
        train_hidden = lstm.extract_hidden(torch.tensor(X_train_seq, dtype=torch.float32, device=device)).cpu().numpy()
        val_hidden = lstm.extract_hidden(torch.tensor(X_val_seq, dtype=torch.float32, device=device)).cpu().numpy()
        
    logger.info("Training AttackTypeClassifier...")
    attack_classifier = AttackTypeClassifier.from_config(
        input_dim=train_hidden.shape[1], cfg=model_cfg.attack_classifier
    )
    class_weights = MultiClassFocalLoss.compute_class_weights(
        train_attack_labels_seq, num_classes=model_cfg.attack_classifier.num_classes, max_ratio=10.0
    ).to(device)
    logger.info("Attack classifier class weights: %s", class_weights.cpu().tolist())
    mc_focal_loss = MultiClassFocalLoss(
        alpha=model_cfg.attack_classifier.focal_alpha,
        gamma=model_cfg.attack_classifier.focal_gamma,
        class_weights=class_weights,
    )
    attack_history = train_attack_classifier(
        attack_classifier, train_hidden, train_attack_labels_seq, val_hidden, val_attack_labels_seq,
        mc_focal_loss, training_cfg,
    )
    save_model(
        attack_classifier,
        models_dir / "attack_classifier.pt",
        metadata={"input_dim": train_hidden.shape[1], "config": "attack_classifier"},
    )
    
    logger.info("Executing calibration via run_stage_calibrate...")
    calibration_summary = run_stage_calibrate(config)

    elapsed = time.perf_counter() - start
    logger.info("Model training completed in %.2f seconds.", elapsed)

    history = {"autoencoder": ae_history, "lstm": lstm_history}
    history_path = models_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training histories saved to %s", history_path)

    return {
        "autoencoder": autoencoder,
        "lstm": lstm,
        "thresholds": thresholds,
        "calibration_summary": calibration_summary,
        "ae_history": ae_history,
        "lstm_history": lstm_history,
    }

def run_stage_infer(
    config: ProjectConfig,
    risk_threshold: float = 0.5,
) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("STAGE 4: Inference & Alert Generation")
    logger.info("=" * 60)
    start = time.perf_counter()

    pipeline = InferencePipeline.from_saved_models(config, risk_threshold)
    results = pipeline.run(split="test", risk_threshold=risk_threshold)

    alerts_df = results["alerts_df"]
    if not alerts_df.empty:
        alerts_path = config.paths.outputs_dir / "alerts.csv"
        alerts_df.to_csv(alerts_path, index=False)
        logger.info("Saved %d alerts to %s", len(alerts_df), alerts_path)

    elapsed = time.perf_counter() - start
    logger.info("Inference completed in %.2f seconds.", elapsed)

    return results

def _budgeted_macro_selection(scores: np.ndarray, y: np.ndarray, attack_types: np.ndarray, budget: int) -> tuple[float, float]:
    candidates = np.arange(0.05, 0.81, 0.01)
    best_threshold, best_score = float(candidates[-1]), -1.0
    benign = attack_types == "none"
    for threshold in candidates:
        pred = scores >= threshold
        if int(pred.sum()) > budget:
            continue
        f1s = []
        for attack in ATTACK_TYPES:
            mask = benign | (attack_types == attack)
            if not (attack_types == attack).any():
                continue
            tp = int((pred[mask] & (attack_types[mask] == attack)).sum())
            fp = int((pred[mask] & benign[mask]).sum())
            fn = int((~pred[mask] & (attack_types[mask] == attack)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        value = float(np.mean(f1s)) if f1s else -1.0
        if value > best_score or (np.isclose(value, best_score) and threshold > best_threshold):
            best_threshold, best_score = float(threshold), value
    if best_score < 0:
        raise RuntimeError(f"No threshold met alert budget {budget}; extend the threshold grid.")
    return best_threshold, best_score

def run_stage_calibrate(config: ProjectConfig, risk_threshold: float = 0.5) -> Dict[str, Any]:
    logger.info("STAGE: Calibration (saved models only)")
    pipeline = InferencePipeline.from_saved_models(config, risk_threshold)
    results = pipeline.run(split="val")
    _, y, entity_types, _, _, _, _, _, attack_types, _ = pipeline._load_split_data("val", config.paths, config.features)
    offset = pipeline.sequence_length - 1
    y, entity_types, attack_types = y[offset:], np.asarray(entity_types[offset:]), np.asarray(attack_types[offset:])
    ae, lstm, policy = results["ae_errors"], results["lstm_probs"], results["policy_scores"]
    budget = config.training.alert_budget
    best_weights, best_threshold, best_value = None, 0.5, -1.0
    for w_ae in np.arange(0.10, 0.81, 0.10):
        for w_lstm in np.arange(0.10, 0.81, 0.10):
            w_policy = round(1.0 - w_ae - w_lstm, 10)
            if w_policy < 0.10:
                continue
            pipeline.risk_scorer.update_weights(float(w_ae), float(w_lstm), w_policy)
            scores = pipeline.risk_scorer.compute(ae, lstm, policy, pipeline.ae_min, pipeline.ae_max)
            threshold, value = _budgeted_macro_selection(scores, y, attack_types, budget)
            if value > best_value:
                best_weights, best_threshold, best_value = (float(w_ae), float(w_lstm), w_policy), threshold, value
    assert best_weights is not None
    pipeline.risk_scorer.update_weights(*best_weights)
    scores = pipeline.risk_scorer.compute(ae, lstm, policy, pipeline.ae_min, pipeline.ae_max)
    quotas = {etype: max(1, round(budget * float((entity_types == etype).mean()))) for etype in ENTITY_TYPES}
    thresholds = {"global": best_threshold}
    for etype in ENTITY_TYPES:
        mask = entity_types == etype
        thresholds[etype], _ = _budgeted_macro_selection(scores[mask], y[mask], attack_types[mask], quotas[etype])
    unscaled = pipeline.scaler.inverse_transform(np.load(config.paths.processed_data_dir / config.features.x_val_file)[offset:])
    feature_df = pd.DataFrame(unscaled, columns=pipeline.feature_names)
    _, _, _, entity_ids, timestamps, source_ips, _, _, _, _ = pipeline._load_split_data("val", config.paths, config.features)
    feature_df["source_ip"] = np.asarray(source_ips[offset:])
    feature_df["entity_id"] = np.asarray(entity_ids[offset:])
    feature_df["timestamp"] = np.asarray(timestamps[offset:])
    signature_thresholds, signature_evidence = tune_signatures(feature_df, attack_types)
    with open(config.paths.risk_weights_file, "w") as f:
        json.dump({"w_ae": best_weights[0], "w_lstm": best_weights[1], "w_policy": best_weights[2], "ae_min": pipeline.ae_min, "ae_max": pipeline.ae_max}, f, indent=2)
    with open(config.paths.risk_thresholds_file, "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(config.paths.models_dir / "signature_thresholds.json", "w") as f:
        json.dump(signature_thresholds, f, indent=2)
    recalls = {attack: float((scores[attack_types == attack] >= best_threshold).mean()) for attack in ATTACK_TYPES if (attack_types == attack).any()}
    for attack, recall in recalls.items():
        if recall < 0.15:
            logger.warning("Post-hoc ML-only recall below 0.15 for %s: %.3f; signature union is evaluated separately.", attack, recall)
    summary = {"selected_weights": dict(zip(("w_ae", "w_lstm", "w_policy"), best_weights)), "global_threshold": best_threshold, "persona_thresholds": thresholds, "alert_budget": budget, "macro_f1": best_value, "ml_only_recalls": recalls, "signature_validation": signature_evidence, "signature_thresholds": signature_thresholds}
    with open(config.paths.outputs_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    run_stage_infer(config, risk_threshold)
    return summary

def run_full_pipeline(
    config: Optional[ProjectConfig] = None,
    risk_threshold: float = 0.5,
) -> Dict[str, Any]:
    if config is None:
        config = get_project_config()

    total_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("BEHAVIORAL ANOMALY DETECTION — FULL PIPELINE")
    logger.info("=" * 60)

    run_stage_generate(config)
    features = run_stage_features(config)
    training = run_stage_train(config)
    inference = run_stage_infer(config, risk_threshold)

    total_elapsed = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETED in %.2f seconds.", total_elapsed)
    logger.info("=" * 60)

    return {
        "features": features,
        "training": training,
        "inference": inference,
        "total_time_seconds": total_elapsed,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral Anomaly Detection Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  generate  — Run synthetic data generation only
  features  — Run feature engineering only
  train     — Run model training only
  infer     — Run inference and alert generation only
  all       — Run the complete end-to-end pipeline

Examples:
  python pipeline_runner.py --stage all
  python pipeline_runner.py --stage train
  python pipeline_runner.py --stage infer --risk-threshold 0.6
        """,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["generate", "features", "train", "calibrate", "infer", "all"],
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.5,
        help="Risk score threshold for alert generation (default: 0.5).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config = get_project_config()

    if args.stage == "generate":
        run_stage_generate(config)
    elif args.stage == "features":
        run_stage_features(config)
    elif args.stage == "train":
        run_stage_train(config)
    elif args.stage == "calibrate":
        run_stage_calibrate(config, args.risk_threshold)
    elif args.stage == "infer":
        run_stage_infer(config, args.risk_threshold)
    elif args.stage == "all":
        run_full_pipeline(config, args.risk_threshold)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
