"""Alert generation engine and temporal attack chain reconstruction.

Translates hybrid risk scores and signature matches into structured SOC alerts with MITRE ATT&CK mapping,
and reconstructs multi-stage attack chains by correlating alerts per entity within a 1-hour correlation window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ATTACK_TYPES, ENTITY_TYPES, ProjectConfig, get_project_config

logger = logging.getLogger(__name__)

MITRE_ATTACK_MAPPING: Dict[str, Dict[str, Any]] = {
    "brute_force": {
        "tactic": "Credential Access",
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "sub_techniques": ["T1110.001", "T1110.003"],
        "severity_base": 0.7,
    },
    "credential_stuffing": {
        "tactic": "Credential Access",
        "technique_id": "T1110.004",
        "technique_name": "Credential Stuffing",
        "sub_techniques": [],
        "severity_base": 0.75,
    },
    "impossible_travel": {
        "tactic": "Initial Access",
        "technique_id": "T1078",
        "technique_name": "Valid Accounts (Impossible Travel)",
        "sub_techniques": ["T1078.004"],
        "severity_base": 0.8,
    },
    "lateral_movement": {
        "tactic": "Lateral Movement",
        "technique_id": "T1021",
        "technique_name": "Remote Services",
        "sub_techniques": ["T1021.002", "T1021.006"],
        "severity_base": 0.85,
    },
    "device_spoofing": {
        "tactic": "Defense Evasion",
        "technique_id": "T1036",
        "technique_name": "Masquerading",
        "sub_techniques": ["T1036.005"],
        "severity_base": 0.65,
    },
    "low_and_slow_exfiltration": {
        "tactic": "Exfiltration",
        "technique_id": "T1048",
        "technique_name": "Exfiltration Over Alternative Protocol",
        "sub_techniques": ["T1048.002"],
        "severity_base": 0.9,
    },
    "insider_drift": {
        "tactic": "Collection",
        "technique_id": "T1074",
        "technique_name": "Data Staged (Insider Threat)",
        "sub_techniques": ["T1074.001"],
        "severity_base": 0.6,
    },
    "unknown": {
        "tactic": "Unknown",
        "technique_id": "N/A",
        "technique_name": "Anomalous Behaviour",
        "sub_techniques": [],
        "severity_base": 0.5,
    },
}

SEVERITY_LEVELS: Dict[str, Tuple[float, float]] = {
    "critical": (0.85, 1.0),
    "high": (0.65, 0.85),
    "medium": (0.45, 0.65),
    "low": (0.25, 0.45),
    "info": (0.0, 0.25),
}

def _classify_severity(risk_score: float) -> str:
    for level, (lo, hi) in SEVERITY_LEVELS.items():
        if lo <= risk_score < hi:
            return level
    return "critical" if risk_score >= 0.85 else "info"

@dataclass
class Alert:

    alert_id: str
    entity_id: str
    entity_type: str
    timestamp: str
    risk_score: float
    severity: str
    ae_error: float
    lstm_prob: float
    policy_score: float
    predicted_attack_type: str
    mitre_tactic: str
    mitre_technique_id: str
    mitre_technique_name: str
    mitre_sub_techniques: List[str]
    source_ip: str = ""
    geo_location: str = ""
    resource_accessed: str = ""
    attack_chain_id: Optional[str] = None
    cold_start: bool = False
    detection_source: str = "ml_threshold"
    classification_confidence: Optional[float] = None
    contributing_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "timestamp": self.timestamp,
            "risk_score": round(self.risk_score, 6),
            "severity": self.severity,
            "ae_error": round(self.ae_error, 6),
            "lstm_prob": round(self.lstm_prob, 6),
            "policy_score": round(self.policy_score, 6),
            "predicted_attack_type": self.predicted_attack_type,
            "mitre_tactic": self.mitre_tactic,
            "mitre_technique_id": self.mitre_technique_id,
            "mitre_technique_name": self.mitre_technique_name,
            "mitre_sub_techniques": ";".join(self.mitre_sub_techniques),
            "source_ip": self.source_ip,
            "geo_location": self.geo_location,
            "resource_accessed": self.resource_accessed,
            "attack_chain_id": self.attack_chain_id or "",
            "cold_start": self.cold_start,
            "detection_source": self.detection_source,
            "classification_confidence": (
                round(float(self.classification_confidence), 6)
                if self.classification_confidence is not None else None
            ),
            "contributing_factors": ";".join(self.contributing_factors) if self.contributing_factors else "",
        }

def _predict_attack_type(
    ae_error: float,
    lstm_prob: float,
    policy_score: float,
    features: Optional[Dict[str, float]] = None,
) -> str:
    if features is None:
        features = {}

    travel_speed = features.get("travel_speed_kmh", 0.0)
    geo_dist = features.get("geo_distance_km", 0.0)
    failed_logins = features.get("failed_logins_last_10min", 0.0)
    fp_consistency = features.get("fingerprint_consistency", 1.0)
    os_mismatch = features.get("os_mismatch", 0.0)
    browser_mismatch = features.get("browser_mismatch", 0.0)
    seq_entropy = features.get("sequence_entropy", 0.0)
    priv_esc = features.get("privilege_escalation_count", 0.0)
    resource_rarity = features.get("resource_rarity", 0.0)
    session_dur = features.get("session_duration", 0.0)
    login_hour = features.get("login_hour", 12.0)

    if failed_logins > 5:
        if geo_dist > 1000:
            return "credential_stuffing"
        return "brute_force"

    if travel_speed > 900.0:
        return "impossible_travel"

    if fp_consistency < 0.5 or os_mismatch == 1.0 or browser_mismatch == 1.0:
        return "device_spoofing"

    if priv_esc > 2 and seq_entropy > 2.0:
        return "lateral_movement"

    if session_dur > 300 and resource_rarity > 0.8:
        return "low_and_slow_exfiltration"

    if (login_hour >= 22 or login_hour <= 5) and lstm_prob > 0.3:
        return "insider_drift"

    if lstm_prob > 0.5:
        return "brute_force"

    return "unknown"

class AlertEngine:

    def __init__(
        self,
        risk_threshold: float = 0.5,
        chain_window_seconds: int = 3600,
        chain_min_alerts: int = 3,
    ) -> None:
        self.risk_threshold: float = risk_threshold
        self.chain_window_seconds: int = chain_window_seconds
        self.chain_min_alerts: int = chain_min_alerts
        logger.info(
            "AlertEngine initialised — threshold=%.2f, chain_window=%ds, chain_min=%d",
            risk_threshold, chain_window_seconds, chain_min_alerts,
        )

    def generate_alerts(
        self,
        risk_scores: np.ndarray,
        ae_errors: np.ndarray,
        lstm_probs: np.ndarray,
        policy_scores: np.ndarray,
        entity_ids: np.ndarray,
        entity_types: np.ndarray,
        timestamps: np.ndarray,
        source_ips: Optional[np.ndarray] = None,
        geo_locations: Optional[np.ndarray] = None,
        resources: Optional[np.ndarray] = None,
        feature_matrix: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        risk_thresholds: Optional[Dict[str, float] | float] = None,
        cold_starts: Optional[np.ndarray] = None,
        contributing_factors: Optional[List[List[str]]] = None,
        predicted_attack_types: Optional[np.ndarray] = None,
        attack_probabilities: Optional[np.ndarray] = None,
        attack_type_to_index: Optional[Dict[str, int]] = None,
        signature_matches: Optional[Dict[str, np.ndarray]] = None,
        signature_factors: Optional[List[List[str]]] = None,
    ) -> pd.DataFrame:
        alerts: List[Dict[str, Any]] = []
        alert_counter = 0

        for i in range(len(risk_scores)):
            etype = str(entity_types[i])
            if isinstance(risk_thresholds, dict):
                t = risk_thresholds.get(etype, risk_thresholds.get("global", self.risk_threshold))
            elif isinstance(risk_thresholds, (float, int)):
                t = float(risk_thresholds)
            else:
                t = self.risk_threshold
                
            ml_triggered = risk_scores[i] >= t
            matching_signatures = [name for name, mask in (signature_matches or {}).items() if bool(mask[i])]
            signature_triggered = bool(matching_signatures)
            if not ml_triggered and not signature_triggered:
                continue

            alert_counter += 1
            alert_id = f"ALR-{alert_counter:06d}"

            feat_dict: Optional[Dict[str, float]] = None
            if feature_matrix is not None and feature_names is not None:
                feat_dict = {
                    name: float(feature_matrix[i, j])
                    for j, name in enumerate(feature_names)
                }

            prediction_from_signature = bool(matching_signatures)
            prediction_from_classifier = False
            if "impossible_travel_signature" in matching_signatures:
                predicted_type = "impossible_travel"
            elif "device_spoofing_signature" in matching_signatures:
                predicted_type = "device_spoofing"
            elif "credential_stuffing_signature" in matching_signatures:
                predicted_type = "credential_stuffing"
            elif predicted_attack_types is not None and predicted_attack_types[i] is not None and str(predicted_attack_types[i]) not in ("unknown", "benign", "None"):
                predicted_type = str(predicted_attack_types[i])
                prediction_from_classifier = True
            else:
                predicted_type = _predict_attack_type(
                    ae_errors[i], lstm_probs[i], policy_scores[i], feat_dict
                )

            mitre = MITRE_ATTACK_MAPPING.get(
                predicted_type, MITRE_ATTACK_MAPPING["unknown"]
            )
            confidence: Optional[float] = None
            if (
                prediction_from_classifier
                and not prediction_from_signature
                and attack_probabilities is not None
                and attack_type_to_index is not None
            ):
                class_index = attack_type_to_index.get(predicted_type)
                if class_index is not None and i < len(attack_probabilities):
                    confidence = float(attack_probabilities[i, class_index])

            severity = _classify_severity(risk_scores[i])

            alert = Alert(
                alert_id=alert_id,
                entity_id=str(entity_ids[i]),
                entity_type=str(entity_types[i]),
                timestamp=str(timestamps[i]),
                risk_score=float(risk_scores[i]),
                severity=severity,
                ae_error=float(ae_errors[i]),
                lstm_prob=float(lstm_probs[i]),
                policy_score=float(policy_scores[i]),
                predicted_attack_type=predicted_type,
                mitre_tactic=mitre["tactic"],
                mitre_technique_id=mitre["technique_id"],
                mitre_technique_name=mitre["technique_name"],
                mitre_sub_techniques=mitre["sub_techniques"],
                source_ip=str(source_ips[i]) if source_ips is not None else "",
                geo_location=str(geo_locations[i]) if geo_locations is not None else "",
                resource_accessed=str(resources[i]) if resources is not None else "",
                cold_start=bool(cold_starts[i]) if cold_starts is not None else False,
                detection_source="both" if ml_triggered and signature_triggered else ("signature_rule" if signature_triggered else "ml_threshold"),
                classification_confidence=confidence,
                contributing_factors=(signature_factors[i] if signature_triggered and signature_factors is not None else contributing_factors[i]) if contributing_factors is not None and len(contributing_factors) > i else [],
            )
            alerts.append(alert.to_dict())

        if not alerts:
            logger.info("No alerts generated (all risk scores below threshold).")
            return pd.DataFrame(columns=[
                "alert_id", "entity_id", "entity_type", "timestamp",
                "risk_score", "severity", "ae_error", "lstm_prob",
                "policy_score", "predicted_attack_type", "mitre_tactic",
                "mitre_technique_id", "mitre_technique_name",
                "mitre_sub_techniques", "source_ip", "geo_location",
                "resource_accessed", "attack_chain_id", "cold_start", "detection_source",
                "classification_confidence", "contributing_factors",
            ])

        alerts_df = pd.DataFrame(alerts)
        alerts_df = alerts_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

        alerts_df = self._reconstruct_chains(alerts_df)

        logger.info(
            "Generated %d alerts (%d critical, %d high, %d medium, %d low, %d info)",
            len(alerts_df),
            (alerts_df["severity"] == "critical").sum(),
            (alerts_df["severity"] == "high").sum(),
            (alerts_df["severity"] == "medium").sum(),
            (alerts_df["severity"] == "low").sum(),
            (alerts_df["severity"] == "info").sum(),
        )

        return alerts_df

    def _reconstruct_chains(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        if alerts_df.empty:
            return alerts_df

        alerts_df = alerts_df.copy()
        alerts_df["_ts_parsed"] = pd.to_datetime(alerts_df["timestamp"], utc=True)
        alerts_df = alerts_df.sort_values(["entity_id", "_ts_parsed"])

        chain_counter = 0
        chain_ids: List[Optional[str]] = [None] * len(alerts_df)
        window = timedelta(seconds=self.chain_window_seconds)

        for entity_id, group in alerts_df.groupby("entity_id"):
            if len(group) < self.chain_min_alerts:
                continue

            group_sorted = group.sort_values("_ts_parsed")
            chain_start_idx = 0
            current_chain: List[int] = [group_sorted.index[0]]

            for j in range(1, len(group_sorted)):
                curr_ts = group_sorted.iloc[j]["_ts_parsed"]
                prev_ts = group_sorted.iloc[j - 1]["_ts_parsed"]

                if (curr_ts - prev_ts) <= window:
                    current_chain.append(group_sorted.index[j])
                else:
                    if len(current_chain) >= self.chain_min_alerts:
                        chain_counter += 1
                        chain_id = f"CHAIN-{chain_counter:04d}"
                        for idx in current_chain:
                            chain_ids[alerts_df.index.get_loc(idx)] = chain_id
                    current_chain = [group_sorted.index[j]]

            if len(current_chain) >= self.chain_min_alerts:
                chain_counter += 1
                chain_id = f"CHAIN-{chain_counter:04d}"
                for idx in current_chain:
                    chain_ids[alerts_df.index.get_loc(idx)] = chain_id

        alerts_df["attack_chain_id"] = chain_ids
        alerts_df = alerts_df.drop(columns=["_ts_parsed"])

        logger.info(
            "Reconstructed %d attack chains from %d alerts.",
            chain_counter, len(alerts_df),
        )
        return alerts_df

    @classmethod
    def from_config(
        cls,
        config: Optional[ProjectConfig] = None,
        risk_threshold: float = 0.5,
    ) -> "AlertEngine":
        return cls(risk_threshold=risk_threshold)

def get_mitre_mapping() -> Dict[str, Dict[str, Any]]:
    return MITRE_ATTACK_MAPPING.copy()

def get_severity_distribution(alerts_df: pd.DataFrame) -> Dict[str, int]:
    if alerts_df.empty:
        return {level: 0 for level in SEVERITY_LEVELS}
    return alerts_df["severity"].value_counts().to_dict()
