"""End-to-end inference orchestrator and Captum Integrated Gradients explainability pipeline.

Loads trained models, executes AE/LSTM/Policy inference, evaluates signature rules, generates alerts,
computes feature attribution via Captum IG, and enforces alert count consistency assertions.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import (
    ENTITY_TYPES,
    NUM_ENTITY_TYPES,
    ATTACK_TYPE_LABELS,
    ProjectConfig,
    get_project_config,
)
from models import (
    FocalLoss,
    LSTMSequenceClassifier,
    PolicyEngine,
    RiskScorer,
    SharedAutoencoder,
    AttackTypeClassifier,
    create_sequences,
    infer_autoencoder,
    infer_lstm,
    load_model,
    load_thresholds,
    set_torch_seed,
    explain_autoencoder,
)
from models import _get_device
from alert_engine import AlertEngine
from evaluation import (
    EvaluationResult,
    LatencyTracker,
    evaluate_predictions,
    find_optimal_threshold,
    generate_report,
)
from drift_monitor import DriftMonitor
from signature_rules import evaluate_signatures

logger = logging.getLogger(__name__)

class InferencePipeline:

    def __init__(
        self,
        autoencoder: SharedAutoencoder,
        lstm_classifier: LSTMSequenceClassifier,
        policy_engine: PolicyEngine,
        risk_scorer: RiskScorer,
        alert_engine: AlertEngine,
        thresholds: Dict[str, float],
        scaler: Any,
        encoder: Any,
        feature_names: List[str],
        device: torch.device,
        sequence_length: int,
        config: ProjectConfig,
        risk_thresholds: Optional[Dict[str, float]] = None,
        platt_scaler: Optional[Any] = None,
        ae_min: float = 0.0,
        ae_max: float = 1.0,
    ) -> None:
        self.autoencoder = autoencoder
        self.lstm_classifier = lstm_classifier
        self.policy_engine = policy_engine
        self.risk_scorer = risk_scorer
        self.alert_engine = alert_engine
        self.thresholds = thresholds
        self.scaler = scaler
        self.encoder = encoder
        self.feature_names = feature_names
        self.device = device
        self.sequence_length = sequence_length
        self.config = config
        self.risk_thresholds = risk_thresholds
        self.platt_scaler = platt_scaler
        self.ae_min = ae_min
        self.ae_max = ae_max
        self.attack_classifier: Optional[AttackTypeClassifier] = None

    def run(
        self,
        X: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        entity_types: Optional[np.ndarray] = None,
        entity_ids: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        source_ips: Optional[np.ndarray] = None,
        geo_locations: Optional[np.ndarray] = None,
        resources: Optional[np.ndarray] = None,
        attack_types: Optional[np.ndarray] = None,
        risk_threshold: float = 0.5,
        split: str = "test",
    ) -> Dict[str, Any]:
        paths = self.config.paths
        feature_cfg = self.config.features
        latency_tracker = LatencyTracker()

        if X is None:
            X, y_true, entity_types, entity_ids, timestamps, source_ips, \
                geo_locations, resources, attack_types, cold_starts = self._load_split_data(
                    split, paths, feature_cfg
                )

        n_samples = len(X)
        logger.info("Running inference on %d samples (split=%s)...", n_samples, split)

        with latency_tracker:
            ae_errors = infer_autoencoder(self.autoencoder, X, self.device)
        logger.info("AE inference complete. Mean error: %.6f", ae_errors.mean())

        drift_results = None
        ref_path = paths.models_dir / "ae_reference_stats.npz"
        if ref_path.exists() and entity_types is not None:
            ref_data = np.load(ref_path, allow_pickle=True)
            ref_errors = ref_data["errors"]
            ref_entity_types = ref_data["entity_types"]
            
            monitor = DriftMonitor(self.config)
            drift_results = monitor.check_drift(ae_errors, entity_types, ref_errors, ref_entity_types)
            
            drift_dict = {k: v.__dict__ for k, v in drift_results.items()}
            with open(paths.outputs_dir / "drift_report.json", "w") as f:
                json.dump(drift_dict, f, indent=2)
            logger.info("Drift check complete. Results saved to drift_report.json")

        dummy_y = np.zeros(n_samples, dtype=np.float64)
        X_seq, y_seq = create_sequences(X, dummy_y, self.sequence_length)

        with latency_tracker:
            self.lstm_classifier.eval()
            X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.lstm_classifier(X_seq_tensor).cpu().numpy().reshape(-1)
            
            if hasattr(self, "platt_scaler") and self.platt_scaler is not None:
                lstm_probs = self.platt_scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
            else:
                lstm_probs = 1.0 / (1.0 + np.exp(-logits))
                
        logger.info("LSTM inference complete. Mean prob: %.6f", lstm_probs.mean())
        
        predicted_attack_types = None
        attack_probs: Optional[np.ndarray] = None
        attack_type_to_index: Optional[Dict[str, int]] = None
        if getattr(self, "attack_classifier", None) is not None:
            with latency_tracker:
                self.attack_classifier.eval()
                self.lstm_classifier.eval()
                with torch.no_grad():
                    hidden_states = self.lstm_classifier.extract_hidden(X_seq_tensor)
                    attack_logits = self.attack_classifier(hidden_states)
                    attack_probs = torch.softmax(attack_logits, dim=1).cpu().numpy()
                    
                idx_to_attack = {v: k for k, v in ATTACK_TYPE_LABELS.items()}
                attack_type_to_index = {attack: index for index, attack in idx_to_attack.items()}
                hard_classes = {"impossible_travel", "device_spoofing", "insider_drift", "low_and_slow_exfiltration"}
                
                predicted_attack_types = []
                for i in range(len(attack_probs)):
                    pred_idx = int(np.argmax(attack_probs[i, 1:])) + 1
                    prob = float(attack_probs[i, pred_idx])
                    atype = idx_to_attack.get(pred_idx, "unknown")
                    threshold = 0.20 if atype in hard_classes else 0.50

                    if prob < threshold:
                        predicted_attack_types.append(None)
                    else:
                        predicted_attack_types.append(atype)
                        
                predicted_attack_types = np.array(predicted_attack_types)
            logger.info("AttackTypeClassifier inference complete.")

        offset = self.sequence_length - 1
        aligned_X = X[offset:]
        
        unscaled_X_all = self.scaler.inverse_transform(aligned_X)

        with latency_tracker:
            feature_dict: Dict[str, np.ndarray] = {}
            for j, name in enumerate(self.feature_names):
                feature_dict[name] = unscaled_X_all[:, j]
            policy_scores = self.policy_engine.evaluate(feature_dict)
        logger.info("Policy engine complete. Mean score: %.6f", policy_scores.mean())

        ae_errors_aligned = ae_errors[offset:]
        min_len = min(len(ae_errors_aligned), len(lstm_probs), len(policy_scores))
        ae_errors_aligned = ae_errors_aligned[:min_len]
        lstm_probs = lstm_probs[:min_len]
        policy_scores = policy_scores[:min_len]
        unscaled_X_aligned = unscaled_X_all[:min_len]
        aligned_cold_starts = cold_starts[offset:offset + min_len] if cold_starts is not None else np.zeros(min_len, dtype=bool)

        with latency_tracker:
            risk_scores = self.risk_scorer.compute(
                ae_errors_aligned, lstm_probs, policy_scores,
                ae_min=getattr(self, "ae_min", 0.0),
                ae_max=getattr(self, "ae_max", 1.0)
            )

            if aligned_cold_starts.any():
                cold_idx = np.where(aligned_cold_starts)[0]
                t_cfg = self.config.training
                orig_weights = (self.risk_scorer.w_ae, self.risk_scorer.w_lstm, self.risk_scorer.w_policy)
                self.risk_scorer.w_ae = t_cfg.cold_start_w_ae
                self.risk_scorer.w_lstm = t_cfg.cold_start_w_lstm
                self.risk_scorer.w_policy = t_cfg.cold_start_w_policy
                
                cold_scores = self.risk_scorer.compute(
                    ae_errors_aligned[cold_idx], 
                    lstm_probs[cold_idx], 
                    policy_scores[cold_idx],
                    ae_min=getattr(self, "ae_min", 0.0),
                    ae_max=getattr(self, "ae_max", 1.0)
                )
                risk_scores[cold_idx] = cold_scores
                
                self.risk_scorer.w_ae, self.risk_scorer.w_lstm, self.risk_scorer.w_policy = orig_weights

        logger.info(
            "Risk scoring complete. Mean: %.4f, Max: %.4f (Cold-starts: %d)",
            risk_scores.mean(), risk_scores.max(), aligned_cold_starts.sum()
        )

        aligned_entity_ids = entity_ids[offset:offset + min_len] if entity_ids is not None else np.array(["unknown"] * min_len)
        aligned_entity_types = entity_types[offset:offset + min_len] if entity_types is not None else np.array(["unknown"] * min_len)
        aligned_timestamps = timestamps[offset:offset + min_len] if timestamps is not None else np.array([""] * min_len)
        aligned_source_ips = source_ips[offset:offset + min_len] if source_ips is not None else None
        aligned_geo_locs = geo_locations[offset:offset + min_len] if geo_locations is not None else None
        aligned_resources = resources[offset:offset + min_len] if resources is not None else None

        signature_thresholds_path = paths.models_dir / "signature_thresholds.json"
        signature_matches: Dict[str, np.ndarray] = {}
        if signature_thresholds_path.exists():
            with open(signature_thresholds_path, "r") as f:
                signature_thresholds = json.load(f)
            signature_features = pd.DataFrame(unscaled_X_aligned, columns=self.feature_names)
            if aligned_source_ips is not None:
                signature_features["source_ip"] = aligned_source_ips
            signature_features["entity_id"] = aligned_entity_ids
            signature_features["timestamp"] = aligned_timestamps
            signature_matches = evaluate_signatures(signature_features, signature_thresholds)
        use_thresholds = getattr(self, "risk_thresholds", None) or risk_threshold
        if isinstance(use_thresholds, dict):
            ml_alert_mask = np.fromiter(
                (risk_scores[i] >= use_thresholds.get(str(aligned_entity_types[i]), use_thresholds.get("global", risk_threshold)) for i in range(min_len)),
                dtype=bool, count=min_len,
            )
        else:
            ml_alert_mask = risk_scores >= float(use_thresholds)
        signature_alert_mask = np.zeros(min_len, dtype=bool)
        for mask in signature_matches.values():
            signature_alert_mask |= mask
        combined_alert_mask = ml_alert_mask | signature_alert_mask
        signature_factors = [[] for _ in range(min_len)]
        for i in np.where(signature_alert_mask)[0]:
            parts = []
            if signature_matches.get("impossible_travel_signature", np.zeros(min_len, dtype=bool))[i]:
                parts.append(f"travel_speed_kmh = {unscaled_X_aligned[i, self.feature_names.index('travel_speed_kmh')]:.0f} km/h (physically implausible)")
            if signature_matches.get("device_spoofing_signature", np.zeros(min_len, dtype=bool))[i]:
                parts.append("device fingerprint mismatch with OS/browser change")
            if signature_matches.get("credential_stuffing_signature", np.zeros(min_len, dtype=bool))[i]:
                parts.append("high failed-login rate across multiple accounts from few source IPs")
            signature_factors[i] = parts

        contributing_factors = [[] for _ in range(min_len)]
        with latency_tracker:
            top_indices = np.where(ml_alert_mask)[0]
            top_k = len(top_indices)
            
            X_top = aligned_X[:min_len][top_indices]
            
            try:
                ig_results = explain_autoencoder(
                    self.autoencoder, X_top, self.feature_names, self.device, n_steps=20
                )
                attrs = ig_results["attributions"]
                
                feature_desc = {
                    "geo_distance_km": ("unusual geographic jump", "low geo distance"),
                    "travel_speed_kmh": ("impossible travel speed", "normal travel speed"),
                    "failed_logins_last_10min": ("high failed logins", "normal failed logins"),
                    "fingerprint_consistency": ("normal fingerprint consistency", "device fingerprint mismatch"),
                    "os_mismatch": ("OS mismatch", "OS matches baseline"),
                    "browser_mismatch": ("browser mismatch", "browser matches baseline"),
                    "sequence_entropy": ("highly anomalous sequence", "normal sequence entropy"),
                    "privilege_escalation_count": ("privilege escalation detected", "normal privilege usage"),
                    "resource_rarity": ("access to rare resource", "access to common resource"),
                    "session_duration": ("unusually long session", "unusually short session"),
                    "login_hour": ("unusual login hour", "typical login hour"),
                }
                
                for i, original_idx in enumerate(top_indices):
                    abs_attrs = np.abs(attrs[i])
                    sorted_indices = np.argsort(abs_attrs)[::-1]
                    
                    filtered_top_3 = []
                    for f_idx in sorted_indices:
                        feat_name = self.feature_names[f_idx]
                        if feat_name.startswith("entity_type_"):
                            continue
                            
                        is_high = X_top[i, f_idx] > 0 
                        
                        if feat_name in feature_desc:
                            phrase = feature_desc[feat_name][0] if is_high else feature_desc[feat_name][1]
                        else:
                            phrase = f"high {feat_name}" if is_high else f"low {feat_name}"
                            
                        filtered_top_3.append(phrase)
                        if len(filtered_top_3) == 3:
                            break
                            
                    contributing_factors[original_idx] = filtered_top_3
                    
                logger.info("Computed Captum IG attributions for top %d samples.", top_k)
            except Exception as e:
                logger.error("Failed to compute IG attributions: %s", e)

        with latency_tracker:
            alerts_df = self.alert_engine.generate_alerts(
                risk_scores=risk_scores,
                ae_errors=ae_errors_aligned,
                lstm_probs=lstm_probs,
                policy_scores=policy_scores,
                entity_ids=aligned_entity_ids,
                entity_types=aligned_entity_types,
                timestamps=aligned_timestamps,
                source_ips=aligned_source_ips,
                geo_locations=aligned_geo_locs,
                resources=aligned_resources,
                feature_matrix=unscaled_X_aligned,
                feature_names=self.feature_names,
                risk_thresholds=use_thresholds,
                cold_starts=aligned_cold_starts,
                contributing_factors=contributing_factors,
                predicted_attack_types=predicted_attack_types[:min_len] if predicted_attack_types is not None else None,
                attack_probabilities=attack_probs[:min_len] if attack_probs is not None else None,
                attack_type_to_index=attack_type_to_index,
                signature_matches=signature_matches,
                signature_factors=signature_factors,
            )
        logger.info("Alert generation complete. %d alerts generated.", len(alerts_df))

        if attack_probs is not None:
            if isinstance(use_thresholds, dict):
                alert_mask = np.fromiter(
                    (risk_scores[i] >= use_thresholds.get(str(aligned_entity_types[i]), use_thresholds.get("global", risk_threshold))
                     for i in range(min_len)),
                    dtype=bool,
                    count=min_len,
                )
            else:
                alert_mask = risk_scores >= float(use_thresholds)
            probability_columns = [f"prob_{idx_to_attack[i]}" for i in range(attack_probs.shape[1])]
            probability_report = pd.DataFrame(attack_probs[:min_len], columns=probability_columns)
            probability_report["true_attack_type"] = (
                attack_types[offset:offset + min_len] if attack_types is not None else "unknown"
            )
            probability_report["predicted_attack_type"] = predicted_attack_types[:min_len]
            probability_report = probability_report.loc[alert_mask].reset_index(drop=True)
            probability_report.to_csv(paths.outputs_dir / "classifier_softmax_alerts.csv", index=False)
            probability_summary = probability_report.groupby("true_attack_type", dropna=False)[probability_columns].mean()
            probability_summary.insert(0, "alert_count", probability_report.groupby("true_attack_type", dropna=False).size())
            probability_summary.to_csv(paths.outputs_dir / "classifier_softmax_summary.csv")

        evaluation: Optional[EvaluationResult] = None
        total_inference_time = latency_tracker.get_latencies().sum() / 1000.0
        if y_true is not None:
            y_true_aligned = y_true[offset:offset + min_len]
            aligned_attack_types = attack_types[offset:offset + min_len] if attack_types is not None else None
            evaluation = evaluate_predictions(
                y_true=y_true_aligned,
                y_scores=risk_scores,
                threshold=use_thresholds,
                predictions=combined_alert_mask.astype(int),
                reconstruction_errors=ae_errors_aligned,
                risk_scores=risk_scores,
                attack_types=aligned_attack_types,
                entity_types=aligned_entity_types,
                latencies_ms=latency_tracker.get_latencies(),
                warmup_calls=3,
                total_inference_time_sec=total_inference_time,
                total_samples_inferred=min_len,
            )
            report_name = "evaluation_report" if split == "test" else f"{split}_evaluation_report"
            generate_report(evaluation, paths.outputs_dir, report_name=report_name)
            logger.info("Evaluation report saved to %s (report_name: %s)", paths.outputs_dir, report_name)

            cm = evaluation.confusion_mat
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            eval_alert_count = fp + tp
            actual_alert_count = len(alerts_df)
            assert actual_alert_count == eval_alert_count, (
                f"LOUD FAILURE: len(alerts.csv) ({actual_alert_count}) "
                f"does not match evaluation confusion matrix FP+TP ({fp}+{tp} = {eval_alert_count})!"
            )
            logger.info("ASSERTION PASSED: len(alerts.csv) (%d) == FP+TP (%d)", actual_alert_count, eval_alert_count)

        return {
            "ae_errors": ae_errors_aligned,
            "lstm_probs": lstm_probs,
            "policy_scores": policy_scores,
            "risk_scores": risk_scores,
            "alerts_df": alerts_df,
            "evaluation": evaluation,
            "latency_tracker": latency_tracker,
            "attack_probabilities": attack_probs[:min_len] if attack_probs is not None else None,
        }

    def _load_split_data(
        self,
        split: str,
        paths: Any,
        feature_cfg: Any,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        proc_dir = paths.processed_data_dir
        file_map = {
            "train": (feature_cfg.x_train_file, feature_cfg.y_train_file),
            "val": (feature_cfg.x_val_file, feature_cfg.y_val_file),
            "test": (feature_cfg.x_test_file, feature_cfg.y_test_file),
        }
        x_file, y_file = file_map[split]
        X = np.load(proc_dir / x_file)
        y = np.load(proc_dir / y_file)

        logger.info("Loaded %s split: X=%s, y=%s", split, X.shape, y.shape)

        df_logs = pd.read_csv(paths.raw_logs_file, parse_dates=["timestamp"])
        df_truth = pd.read_csv(paths.ground_truth_file, parse_dates=["timestamp"])
        df_logs = df_logs.sort_values("timestamp").reset_index(drop=True)
        df_truth = df_truth.sort_values("timestamp").reset_index(drop=True)

        n = len(df_logs)
        train_end = int(n * feature_cfg.train_ratio)
        val_end = train_end + int(n * feature_cfg.val_ratio)

        if split == "train":
            df_slice = df_logs.iloc[:train_end]
            truth_slice = df_truth.iloc[:train_end]
        elif split == "val":
            df_slice = df_logs.iloc[train_end:val_end]
            truth_slice = df_truth.iloc[train_end:val_end]
        else:
            df_slice = df_logs.iloc[val_end:]
            truth_slice = df_truth.iloc[val_end:]

        entity_ids = df_slice["entity_id"].values
        entity_types = df_slice["entity_type"].values
        timestamps = df_slice["timestamp"].astype(str).values
        source_ips = df_slice["source_ip"].values
        geo_locs = df_slice["geo_location"].values
        resources = df_slice["resource_accessed"].values
        attack_types_arr = truth_slice["attack_type"].values

        first_seen = df_logs.groupby("entity_id")["timestamp"].transform("min")
        days_since_first = (df_logs["timestamp"] - first_seen).dt.total_seconds() / 86400.0
        min_days_map = self.config.training.cold_start_min_history_days
        min_days_series = df_logs["entity_type"].map(min_days_map).fillna(0.0)
        df_logs["cold_start"] = days_since_first < min_days_series
        
        if split == "train":
            cold_starts_slice = df_logs.iloc[:train_end]["cold_start"].values
        elif split == "val":
            cold_starts_slice = df_logs.iloc[train_end:val_end]["cold_start"].values
        else:
            cold_starts_slice = df_logs.iloc[val_end:]["cold_start"].values

        return X, y, entity_types, entity_ids, timestamps, source_ips, geo_locs, resources, attack_types_arr, cold_starts_slice

    @classmethod
    def from_saved_models(
        cls,
        config: Optional[ProjectConfig] = None,
        risk_threshold: float = 0.5,
    ) -> "InferencePipeline":
        if config is None:
            config = get_project_config()

        paths = config.paths
        proc_dir = paths.processed_data_dir
        models_dir = paths.models_dir
        model_cfg = config.model
        training_cfg = config.training

        device = _get_device(training_cfg.device)
        set_torch_seed(training_cfg.random_seed)

        metadata_path = proc_dir / config.features.feature_metadata_file
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        feature_names = metadata["feature_names"]
        num_features = metadata["num_features"]

        logger.info("Loaded feature metadata: %d features.", num_features)

        with open(proc_dir / config.features.scaler_file, "rb") as f:
            scaler = pickle.load(f)
        with open(proc_dir / config.features.encoder_file, "rb") as f:
            encoder = pickle.load(f)

        autoencoder = SharedAutoencoder.from_config(
            input_dim=num_features, cfg=model_cfg.autoencoder
        )
        ae_path = models_dir / "autoencoder.pt"
        if ae_path.exists():
            load_model(autoencoder, ae_path, device)
        else:
            logger.warning("Autoencoder model file not found at %s", ae_path)
        autoencoder = autoencoder.to(device)

        lstm = LSTMSequenceClassifier.from_config(
            input_dim=num_features, cfg=model_cfg.lstm
        )
        lstm_path = models_dir / "lstm_classifier.pt"
        if lstm_path.exists():
            load_model(lstm, lstm_path, device)
        else:
            logger.warning("LSTM model file not found at %s", lstm_path)
        lstm = lstm.to(device)

        thresholds_path = models_dir / "thresholds.json"
        if thresholds_path.exists():
            thresholds = load_thresholds(thresholds_path)
        else:
            logger.warning("Thresholds not found. Using default 0.5 for all types.")
            thresholds = {etype: 0.5 for etype in ENTITY_TYPES}

        policy_engine = PolicyEngine.from_config(model_cfg.policy_engine)
        risk_scorer = RiskScorer.from_config(model_cfg.risk_scorer)
        alert_engine = AlertEngine.from_config(config, risk_threshold)
        
        rw_path = models_dir / "risk_weights.json"
        ae_min, ae_max = 0.0, 1.0
        if rw_path.exists():
            with open(rw_path, "r") as f:
                rw = json.load(f)
            risk_scorer.update_weights(rw["w_ae"], rw["w_lstm"], rw["w_policy"])
            ae_min = rw.get("ae_min", 0.0)
            ae_max = rw.get("ae_max", 1.0)
            
        rt_path = models_dir / "risk_thresholds.json"
        risk_thresholds = None
        if rt_path.exists():
            with open(rt_path, "r") as f:
                risk_thresholds = json.load(f)
                
        ps_path = models_dir / "platt_scaler.pkl"
        platt_scaler = None
        if ps_path.exists():
            with open(ps_path, "rb") as f:
                platt_scaler = pickle.load(f)

        pipeline = cls(
            autoencoder=autoencoder,
            lstm_classifier=lstm,
            policy_engine=policy_engine,
            risk_scorer=risk_scorer,
            alert_engine=alert_engine,
            thresholds=thresholds,
            scaler=scaler,
            encoder=encoder,
            feature_names=feature_names,
            device=device,
            sequence_length=model_cfg.lstm.sequence_length,
            config=config,
            risk_thresholds=risk_thresholds,
            platt_scaler=platt_scaler,
            ae_min=ae_min,
            ae_max=ae_max,
        )
        
        ac_path = models_dir / "attack_classifier.pt"
        if ac_path.exists():
            attack_classifier = AttackTypeClassifier.from_config(
                input_dim=model_cfg.lstm.hidden_dim * (2 if model_cfg.lstm.bidirectional else 1),
                cfg=model_cfg.attack_classifier
            )
            load_model(attack_classifier, ac_path, device)
            pipeline.attack_classifier = attack_classifier.to(device)

        logger.info("InferencePipeline initialised from saved models.")
        return pipeline
