"""High-precision, validation-tuned behavioral attack signatures.

Fast-path rules for impossible travel (speed > 600 km/h + external IP), device spoofing (OS/browser mismatch),
and credential stuffing, unioned with the ML detection threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SignatureRule:
    name: str
    condition: Callable[[pd.DataFrame, Dict[str, float]], np.ndarray]
    mitre_technique_id: str
    confidence: float

def impossible_travel_mask(features: pd.DataFrame, thresholds: Dict[str, float]) -> np.ndarray:
    speed = features["travel_speed_kmh"].to_numpy() > thresholds["speed_threshold_kmh"]
    if "source_ip" in features:
        external_ip = features["source_ip"].astype(str).str.startswith("172.16.").to_numpy()
        return speed & external_ip
    return speed

def device_spoofing_mask(features: pd.DataFrame, thresholds: Dict[str, float]) -> np.ndarray:
    mismatch = (features["os_mismatch"].to_numpy() == 1) | (features["browser_mismatch"].to_numpy() == 1)
    return mismatch & (features["fingerprint_consistency"].to_numpy() < thresholds["fingerprint_consistency_threshold"])

SIGNATURE_RULES = (
    SignatureRule("impossible_travel_signature", impossible_travel_mask, "T1078", 0.99),
    SignatureRule("device_spoofing_signature", device_spoofing_mask, "T1036", 0.98),
)

def credential_stuffing_mask(features: pd.DataFrame, thresholds: Dict[str, float]) -> np.ndarray:
    required = {"timestamp", "entity_id", "source_ip", "session_duration"}
    threshold_keys = {"credential_window_seconds", "credential_entity_count", "credential_source_ip_count", "credential_failure_rate"}
    if not required.issubset(features.columns) or not threshold_keys.issubset(thresholds):
        return np.zeros(len(features), dtype=bool)
    ordered = features.copy()
    ordered["_original_index"] = np.arange(len(ordered))
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered = ordered.sort_values("timestamp")
    times = ordered["timestamp"].astype("int64").to_numpy() // 1_000_000_000
    failed = ordered["session_duration"].to_numpy(dtype=float) < 0.5
    entity_values = ordered["entity_id"].astype(str).to_numpy()
    ip_values = ordered["source_ip"].astype(str).to_numpy()
    out = np.zeros(len(ordered), dtype=bool)
    start = 0
    window_seconds = int(thresholds["credential_window_seconds"])
    entity_counts: Dict[str, int] = {}
    ip_counts: Dict[str, int] = {}
    failure_count = 0
    for end in range(len(ordered)):
        entity_counts[entity_values[end]] = entity_counts.get(entity_values[end], 0) + 1
        ip_counts[ip_values[end]] = ip_counts.get(ip_values[end], 0) + 1
        failure_count += int(failed[end])
        while times[end] - times[start] > window_seconds:
            entity_counts[entity_values[start]] -= 1
            if entity_counts[entity_values[start]] == 0:
                del entity_counts[entity_values[start]]
            ip_counts[ip_values[start]] -= 1
            if ip_counts[ip_values[start]] == 0:
                del ip_counts[ip_values[start]]
            failure_count -= int(failed[start])
            start += 1
        failure_rate = failure_count / (end - start + 1)
        out[end] = (
            len(entity_counts) >= int(thresholds["credential_entity_count"])
            and len(ip_counts) <= int(thresholds["credential_source_ip_count"])
            and failure_rate >= float(thresholds["credential_failure_rate"])
        )
    result = np.zeros(len(features), dtype=bool)
    result[ordered["_original_index"].to_numpy()] = out
    return result

SIGNATURE_RULES = SIGNATURE_RULES + (
    SignatureRule("credential_stuffing_signature", credential_stuffing_mask, "T1110.004", 0.97),
)

def tune_signatures(features: pd.DataFrame, attack_types: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
    benign = attack_types == "none"
    impossible = attack_types == "impossible_travel"
    device = attack_types == "device_spoofing"
    evidence: Dict[str, float] = {}

    for speed in range(600, 1501, 50):
        fires = impossible_travel_mask(features, {"speed_threshold_kmh": float(speed)})
        fp = int((fires & benign).sum())
        if fp <= 3:
            speed_threshold = float(speed)
            break
    else:
        speed_threshold = 1500.0
    speed_fires = impossible_travel_mask(features, {"speed_threshold_kmh": speed_threshold})
    evidence.update({
        "impossible_travel_validation_fp": int((speed_fires & benign).sum()),
        "impossible_travel_validation_recall": float(speed_fires[impossible].mean()) if impossible.any() else 0.0,
    })
    credential_candidates = []
    for window in (300, 600, 900):
        for entity_count in (3, 5):
            for source_count in (1, 3):
                for failure_rate in (0.5, 0.9):
                    candidate = {
                        "credential_window_seconds": window,
                        "credential_entity_count": entity_count,
                        "credential_source_ip_count": source_count,
                        "credential_failure_rate": failure_rate,
                    }
                    fires = credential_stuffing_mask(features, candidate)
                    credential = attack_types == "credential_stuffing"
                    credential_candidates.append({**candidate, "validation_fp": int((fires & benign).sum()), "validation_recall": float(fires[credential].mean()) if credential.any() else 0.0})
    feasible = [x for x in credential_candidates if x["validation_fp"] <= 9 and x["validation_recall"] > 0]
    selected = max(feasible, key=lambda x: (x["validation_recall"], x["credential_entity_count"], -x["credential_source_ip_count"], x["credential_failure_rate"])) if feasible else None
    if selected:
        evidence["credential_stuffing_validation"] = {"selected": selected, "candidates": credential_candidates}
    else:
        evidence["credential_stuffing_validation"] = {"selected": None, "candidates": credential_candidates}

    candidates = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        fires = device_spoofing_mask(features, {"fingerprint_consistency_threshold": float(threshold)})
        recall = float(fires[device].mean()) if device.any() else 0.0
        if recall >= 0.60:
            candidates.append((int((fires & benign).sum()), -float(threshold), float(threshold), recall))
    _, _, fingerprint_threshold, device_recall = min(candidates) if candidates else (0, 0, 0.5, 0.0)
    device_fires = device_spoofing_mask(features, {"fingerprint_consistency_threshold": fingerprint_threshold})
    evidence.update({
        "device_spoofing_validation_fp": int((device_fires & benign).sum()),
        "device_spoofing_validation_recall": device_recall,
    })
    thresholds = {
        "speed_threshold_kmh": speed_threshold,
        "fingerprint_consistency_threshold": fingerprint_threshold,
    }
    if selected:
        thresholds.update({key: selected[key] for key in ("credential_window_seconds", "credential_entity_count", "credential_source_ip_count", "credential_failure_rate")})
    else:
        thresholds.update({"credential_window_seconds": 300, "credential_entity_count": 999999, "credential_source_ip_count": 0, "credential_failure_rate": 1.0})
    return thresholds, evidence

def evaluate_signatures(features: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, np.ndarray]:
    return {rule.name: rule.condition(features, thresholds) for rule in SIGNATURE_RULES}
