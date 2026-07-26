"""Automated smoke test suite verifying risk scorer monotonicity, weight sums, alert fallback thresholds,
prediction source confidence, temporal leakage, and recall guardrails.
"""

import pytest
import numpy as np

from config import get_project_config, RiskScorerConfig
from models import RiskScorer

def test_risk_scorer_monotonicity():
    cfg = RiskScorerConfig(w_ae=0.4, w_lstm=0.4, w_policy=0.2)
    scorer = RiskScorer(cfg)
    
    ae_errors = np.array([0.1, 0.5, 0.9])
    lstm_probs = np.array([0.1, 0.5, 0.9])
    policy_scores = np.array([0.1, 0.5, 0.9])
    
    scores = scorer.compute(ae_errors, lstm_probs, policy_scores, ae_min=0.0, ae_max=1.0)
    
    assert scores[0] < scores[1] < scores[2], "Risk scores are not monotonically increasing."
    
def test_risk_scorer_weight_sum():
    cfg = RiskScorerConfig(w_ae=0.5, w_lstm=0.3, w_policy=0.2)
    scorer = RiskScorer(cfg)
    
    ae_errors = np.array([1.0])
    lstm_probs = np.array([1.0])
    policy_scores = np.array([1.0])
    
    scores = scorer.compute(ae_errors, lstm_probs, policy_scores, ae_min=0.0, ae_max=1.0)
    
    assert np.isclose(scores[0], 1.0), "Risk score did not sum correctly."

def test_alert_engine_fallback_threshold(monkeypatch):
    from alert_engine import AlertEngine
    
    engine = AlertEngine(risk_threshold=0.5)
    
    risk_scores = np.array([0.6])
    ae_errors = np.array([0.5])
    lstm_probs = np.array([0.5])
    policy_scores = np.array([0.5])
    entity_ids = np.array(["user_1"])
    entity_types = np.array(["unknown_persona"])
    timestamps = np.array(["2025-01-01 12:00:00"])
    
    risk_thresholds = {"global": 0.55, "corporate_employee": 0.8}
    
    alerts_df = engine.generate_alerts(
        risk_scores=risk_scores,
        ae_errors=ae_errors,
        lstm_probs=lstm_probs,
        policy_scores=policy_scores,
        entity_ids=entity_ids,
        entity_types=entity_types,
        timestamps=timestamps,
        risk_thresholds=risk_thresholds
    )
    
    assert len(alerts_df) == 1, "Fallback to global threshold failed."
    
    risk_scores_below = np.array([0.5])
    alerts_df_below = engine.generate_alerts(
        risk_scores=risk_scores_below,
        ae_errors=ae_errors,
        lstm_probs=lstm_probs,
        policy_scores=policy_scores,
        entity_ids=entity_ids,
        entity_types=entity_types,
        timestamps=timestamps,
        risk_thresholds=risk_thresholds
    )
    
    assert len(alerts_df_below) == 0, "Fallback to global threshold failed (allowed alert below threshold)."

def test_alert_confidence_respects_prediction_source():
    from alert_engine import AlertEngine

    engine = AlertEngine(risk_threshold=0.5)
    base_args = dict(
        risk_scores=np.array([0.9, 0.9]),
        ae_errors=np.array([0.5, 0.5]),
        lstm_probs=np.array([0.9, 0.9]),
        policy_scores=np.array([0.5, 0.5]),
        entity_ids=np.array(["signature_user", "classifier_user"]),
        entity_types=np.array(["corporate_employee", "corporate_employee"]),
        timestamps=np.array(["2025-01-01 12:00:00", "2025-01-01 14:00:00"]),
        predicted_attack_types=np.array(["brute_force", "brute_force"]),
        attack_probabilities=np.array([[0.99, 0.001, 0.009], [0.05, 0.9, 0.05]]),
        attack_type_to_index={"brute_force": 1, "device_spoofing": 2},
        signature_matches={"device_spoofing_signature": np.array([True, False])},
    )

    alerts_df = engine.generate_alerts(**base_args)
    signature_alert = alerts_df.loc[alerts_df["entity_id"] == "signature_user"].iloc[0]
    classifier_alert = alerts_df.loc[alerts_df["entity_id"] == "classifier_user"].iloc[0]

    assert signature_alert["predicted_attack_type"] == "device_spoofing"
    assert np.isnan(signature_alert["classification_confidence"])
    assert classifier_alert["classification_confidence"] == pytest.approx(0.9)

def test_temporal_leakage():
    import pandas as pd
    from feature_pipeline import (
        _compute_behavioral_features,
        _compute_network_features,
        _compute_device_resource_features,
        _compute_relationship_features,
        _encode_entity_type,
        _init_geo_coords
    )
    from config import get_project_config
    
    cfg = get_project_config()
    _init_geo_coords(cfg)
    
    df1 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 10:05:00", "2025-01-01 10:10:00"]),
        "entity_id": ["user1", "user1", "user1"],
        "entity_type": ["corporate_employee"] * 3,
        "source_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.1"],
        "geo_location": ["US", "CA", "US"],
        "resource_accessed": ["res1", "res2", "res1"],
        "action": ["login", "download", "login"],
        "status": ["success", "success", "failure"],
        "device_fingerprint": ["dev1", "dev1", "dev2"],
        "session_duration": [30.0, 45.0, 10.0],
        "command_sequence": ["login", "login;download", "login"],
    })
    
    df2 = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 10:05:00", "2025-01-01 10:10:00", "2025-01-01 10:15:00"]),
        "entity_id": ["user1", "user1", "user1", "user1"],
        "entity_type": ["corporate_employee"] * 4,
        "source_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.1", "10.0.0.99"],
        "geo_location": ["US", "CA", "US", "RU"],
        "resource_accessed": ["res1", "res2", "res1", "res99"],
        "action": ["login", "download", "login", "delete"],
        "status": ["success", "success", "failure", "success"],
        "device_fingerprint": ["dev1", "dev1", "dev2", "dev99"],
        "session_duration": [30.0, 45.0, 10.0, 60.0],
        "command_sequence": ["login", "login;download", "login", "delete;exit"],
    })
    
    def run_feat(df):
        df = df.copy()
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        
        b = _compute_behavioral_features(df)
        n = _compute_network_features(df, cfg.features)
        dr = _compute_device_resource_features(df, cfg.features)
        r = _compute_relationship_features(df)
        return pd.concat([b, n, dr, r], axis=1)

    feat1 = run_feat(df1)
    feat2 = run_feat(df2)
    
    for col in feat1.columns:
        if col in ["timestamp", "entity_id", "entity_type", "source_ip", "geo_location", "resource_accessed", "action", "status", "device_fingerprint"]:
            continue
        np.testing.assert_array_almost_equal(
            feat1[col].values,
            feat2[col].values[:3],
            err_msg=f"Temporal leakage detected in feature '{col}'!"
        )

def test_attack_wise_recall_guardrail():
    import json
    from inference_pipeline import InferencePipeline
    from config import get_project_config
    
    cfg = get_project_config()
    if not cfg.paths.risk_thresholds_file.exists():
        pytest.skip("Models not yet trained, skipping recall guardrail test.")
        
    pipeline = InferencePipeline.from_saved_models(cfg)
    results = pipeline.run(split="val")
    
    evaluation = results.get("evaluation")
    assert evaluation is not None, "Evaluation results missing."
    
    attack_metrics = evaluation.attack_wise_metrics
    for attack_type, metrics in attack_metrics.items():
        if metrics.support > 0:
            assert metrics.recall >= 0.15, f"Guardrail failed: {attack_type} recall is {metrics.recall:.4f} < 0.15"
