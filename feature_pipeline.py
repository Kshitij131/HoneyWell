"""Feature engineering and preprocessing pipeline extracting 19 behavioral, network, device, and relationship features.

Computes sliding window metrics (failed logins, travel speed, resource rarity, sequence entropy, privilege escalation count)
and scales numerical attributes using StandardScaler.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from config import (
    ENTITY_TYPES,
    ENTITY_TYPE_TO_INDEX,
    GROUND_TRUTH_COLUMNS,
    NUM_ENTITY_TYPES,
    RAW_LOG_COLUMNS,
    FeatureConfig,
    PathConfig,
    ProjectConfig,
    get_project_config,
)

logger = logging.getLogger(__name__)

_GEO_COORDS: Dict[str, Tuple[float, float]] = {}

def _init_geo_coords(config: ProjectConfig) -> None:
    global _GEO_COORDS
    _GEO_COORDS = {
        label: (lat, lon)
        for lat, lon, label in config.data_gen.geo_locations
    }
    logger.debug("Geo-coordinate lookup initialised with %d locations.", len(_GEO_COORDS))

def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    earth_radius_km: float,
) -> np.ndarray:
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius_km * c

def _compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    result["login_hour"] = df["timestamp"].dt.hour.astype(np.float64)
    result["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.float64)
    result["session_duration"] = df["session_duration"].astype(np.float64)

    result["time_since_last_login"] = (
        df.groupby("entity_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .astype(np.float64)
    )

    logger.info("Computed behavioural features: %s", list(result.columns))
    return result

def _compute_network_features(
    df: pd.DataFrame,
    feature_cfg: FeatureConfig,
) -> pd.DataFrame:
    n = len(df)
    unique_ips = np.zeros(n, dtype=np.float64)
    geo_dist = np.zeros(n, dtype=np.float64)
    failed_logins = np.zeros(n, dtype=np.float64)
    travel_speed = np.zeros(n, dtype=np.float64)

    ip_window_s = feature_cfg.unique_ip_window_seconds
    fail_window_s = feature_cfg.failed_login_window_seconds
    earth_r = feature_cfg.geo_distance_earth_radius_km

    ts_epoch = df["timestamp"].values.astype("datetime64[s]").astype(np.int64)

    failed_mask = (df["session_duration"].values < 0.5).astype(np.float64)

    for entity_id, group_idx in df.groupby("entity_id").groups.items():
        idx_arr = group_idx.values if hasattr(group_idx, "values") else np.array(group_idx)
        idx_arr = np.sort(idx_arr)

        entity_ts = ts_epoch[idx_arr]
        entity_ips = df["source_ip"].values[idx_arr]
        entity_geo = df["geo_location"].values[idx_arr]

        for j_pos, j in enumerate(idx_arr):
            current_ts = entity_ts[j_pos]

            window_start_ts = current_ts - ip_window_s
            past_mask = (entity_ts[:j_pos + 1] >= window_start_ts)
            past_ips = entity_ips[:j_pos + 1][past_mask]
            unique_ips[j] = float(len(set(past_ips)))

            if j_pos > 0:
                prev_geo_label = entity_geo[j_pos - 1]
                curr_geo_label = entity_geo[j_pos]
                prev_coords = _GEO_COORDS.get(prev_geo_label)
                curr_coords = _GEO_COORDS.get(curr_geo_label)
                if prev_coords is not None and curr_coords is not None:
                    dist = float(
                        _haversine_km(
                            np.array([prev_coords[0]]),
                            np.array([prev_coords[1]]),
                            np.array([curr_coords[0]]),
                            np.array([curr_coords[1]]),
                            earth_r,
                        )[0]
                    )
                    geo_dist[j] = dist
                    time_delta_sec = current_ts - entity_ts[j_pos - 1]
                    time_delta_hours = time_delta_sec / 3600.0
                    travel_speed[j] = dist / max(time_delta_hours, 1e-5)

            fail_window_start = current_ts - fail_window_s
            fail_past_mask = (entity_ts[:j_pos + 1] >= fail_window_start)
            fail_subset_indices = idx_arr[:j_pos + 1][fail_past_mask]
            failed_logins[j] = float(failed_mask[fail_subset_indices].sum())

    result = pd.DataFrame(
        {
            "unique_ips_last_hour": unique_ips,
            "geo_distance_km": geo_dist,
            "failed_logins_last_10min": failed_logins,
            "travel_speed_kmh": travel_speed,
        },
        index=df.index,
    )

    logger.info("Computed network features: %s", list(result.columns))
    return result

def _compute_shannon_entropy(sequence_str: str, base: int) -> float:
    if not sequence_str or not isinstance(sequence_str, str):
        return 0.0
    tokens = sequence_str.split(";")
    if len(tokens) <= 1:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    log_base = math.log(base) if base != math.e else 1.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * (math.log(p) / log_base)
    return entropy

def _compute_device_resource_features(
    df: pd.DataFrame,
    feature_cfg: FeatureConfig,
) -> pd.DataFrame:
    n = len(df)
    fp_consistency = np.zeros(n, dtype=np.float64)
    resource_rarity = np.zeros(n, dtype=np.float64)
    seq_entropy = np.zeros(n, dtype=np.float64)
    priv_esc_count = np.zeros(n, dtype=np.float64)
    os_mismatch = np.zeros(n, dtype=np.float64)
    browser_mismatch = np.zeros(n, dtype=np.float64)

    entropy_base = feature_cfg.sequence_entropy_base

    priv_esc_keywords = frozenset({
        "chmod", "sudo", "su", "runas", "psexec", "mimikatz",
        "pass_the_hash", "net_user", "reg_query", "wmic",
    })

    seq_entropy = (
        df["command_sequence"]
        .apply(lambda s: _compute_shannon_entropy(s, entropy_base))
        .values.astype(np.float64)
    )

    resource_counter: Counter = Counter()
    total_events_so_far: int = 0

    entity_fp_history: Dict[str, List[str]] = {}
    entity_priv_esc: Dict[str, int] = {}
    entity_seen_os: Dict[str, set] = {}
    entity_seen_browser: Dict[str, set] = {}

    for i in range(n):
        entity_id = df.iloc[i]["entity_id"]
        fingerprint = df.iloc[i]["device_fingerprint"]
        resource = df.iloc[i]["resource_accessed"]
        cmd_seq = df.iloc[i]["command_sequence"]

        if entity_id not in entity_fp_history:
            entity_fp_history[entity_id] = []
            entity_seen_os[entity_id] = set()
            entity_seen_browser[entity_id] = set()

        history = entity_fp_history[entity_id]
        if len(history) > 0:
            match_count = sum(1 for fp in history if fp == fingerprint)
            fp_consistency[i] = match_count / len(history)
        else:
            fp_consistency[i] = 1.0

        history.append(fingerprint)

        parts = str(fingerprint).split("-")
        current_os = parts[0] if len(parts) > 0 else "unknown"
        current_browser = parts[1] if len(parts) > 1 else "unknown"
        
        if len(entity_seen_os[entity_id]) > 0:
            if current_os not in entity_seen_os[entity_id]:
                os_mismatch[i] = 1.0
            if current_browser not in entity_seen_browser[entity_id]:
                browser_mismatch[i] = 1.0
                
        entity_seen_os[entity_id].add(current_os)
        entity_seen_browser[entity_id].add(current_browser)

        total_events_so_far += 1
        resource_counter[resource] += 1
        resource_freq = resource_counter[resource] / total_events_so_far
        resource_rarity[i] = 1.0 - resource_freq

        if entity_id not in entity_priv_esc:
            entity_priv_esc[entity_id] = 0
        if isinstance(cmd_seq, str):
            tokens = cmd_seq.split(";")
            esc_count = sum(1 for t in tokens if t.strip() in priv_esc_keywords)
            entity_priv_esc[entity_id] += esc_count
        priv_esc_count[i] = float(entity_priv_esc[entity_id])

    result = pd.DataFrame(
        {
            "fingerprint_consistency": fp_consistency,
            "resource_rarity": resource_rarity,
            "sequence_entropy": seq_entropy,
            "privilege_escalation_count": priv_esc_count,
            "os_mismatch": os_mismatch,
            "browser_mismatch": browser_mismatch,
        },
        index=df.index,
    )

    logger.info("Computed device & resource features: %s", list(result.columns))
    return result

def _compute_relationship_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    degree_centrality = np.zeros(n, dtype=np.float64)
    newly_observed = np.zeros(n, dtype=np.float64)

    graph = nx.Graph()
    entity_neighbors: Dict[str, set] = {}

    for i in range(n):
        entity_id = df.iloc[i]["entity_id"]
        resource = df.iloc[i]["resource_accessed"]

        if entity_id not in entity_neighbors:
            entity_neighbors[entity_id] = set()

        if resource not in entity_neighbors[entity_id]:
            newly_observed[i] = 1.0
            entity_neighbors[entity_id].add(resource)

        graph.add_edge(entity_id, resource)

        num_nodes = graph.number_of_nodes()
        if num_nodes > 1:
            entity_degree = graph.degree(entity_id)
            degree_centrality[i] = entity_degree / (num_nodes - 1)
        else:
            degree_centrality[i] = 0.0

    result = pd.DataFrame(
        {
            "degree_centrality": degree_centrality,
            "newly_observed_neighbors": newly_observed,
        },
        index=df.index,
    )

    logger.info("Computed relationship features: %s", list(result.columns))
    return result

def _encode_entity_type(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    le.fit(list(ENTITY_TYPES))
    encoded = le.transform(df["entity_type"].values)

    one_hot = np.zeros((len(encoded), NUM_ENTITY_TYPES), dtype=np.float64)
    one_hot[np.arange(len(encoded)), encoded] = 1.0

    logger.info(
        "Encoded entity_type — classes: %s, one-hot shape: %s",
        list(le.classes_),
        one_hot.shape,
    )
    return one_hot, le

def _assemble_features(
    entity_type_onehot: np.ndarray,
    behavioral_df: pd.DataFrame,
    network_df: pd.DataFrame,
    device_resource_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    onehot_names = [f"entity_type_{etype}" for etype in ENTITY_TYPES]

    feature_names = (
        onehot_names
        + list(behavioral_df.columns)
        + list(network_df.columns)
        + list(device_resource_df.columns)
        + list(relationship_df.columns)
    )

    feature_matrix = np.hstack(
        [
            entity_type_onehot,
            behavioral_df.values.astype(np.float64),
            network_df.values.astype(np.float64),
            device_resource_df.values.astype(np.float64),
            relationship_df.values.astype(np.float64),
        ]
    )

    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "Assembled feature matrix: shape=%s, features=%d",
        feature_matrix.shape,
        len(feature_names),
    )
    return feature_matrix, feature_names

def _temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_cfg: FeatureConfig,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    n = len(X)
    train_end = int(n * feature_cfg.train_ratio)
    val_end = train_end + int(n * feature_cfg.val_ratio)

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    logger.info(
        "Temporal split — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def _fit_and_transform_scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Fitted and applied %s scaler.", scaler_type)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def _save_artefacts(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    scaler: Any,
    encoder: LabelEncoder,
    paths: PathConfig,
    feature_cfg: FeatureConfig,
) -> None:
    out_dir = paths.processed_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / feature_cfg.x_train_file, X_train)
    np.save(out_dir / feature_cfg.x_val_file, X_val)
    np.save(out_dir / feature_cfg.x_test_file, X_test)

    np.save(out_dir / feature_cfg.y_train_file, y_train)
    np.save(out_dir / feature_cfg.y_val_file, y_val)
    np.save(out_dir / feature_cfg.y_test_file, y_test)

    metadata = {
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "num_entity_type_features": NUM_ENTITY_TYPES,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "scaler_type": feature_cfg.scaler_type,
        "feature_groups": {
            "entity_type_onehot": [
                f"entity_type_{etype}" for etype in ENTITY_TYPES
            ],
            "behavioral": list(feature_cfg.behavioral_features),
            "network": list(feature_cfg.network_features),
            "device_resource": list(feature_cfg.device_resource_features),
            "relationship": list(feature_cfg.relationship_features),
        },
    }
    metadata_path = out_dir / feature_cfg.feature_metadata_file
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    scaler_path = out_dir / feature_cfg.scaler_file
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    encoder_path = out_dir / feature_cfg.encoder_file
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    logger.info("Saved all artefacts to %s", out_dir)
    logger.info(
        "  Feature arrays: X_train=%s, X_val=%s, X_test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    logger.info(
        "  Label arrays: y_train=%s, y_val=%s, y_test=%s",
        y_train.shape,
        y_val.shape,
        y_test.shape,
    )

def run_feature_pipeline(
    config: Optional[ProjectConfig] = None,
) -> Dict[str, Any]:
    if config is None:
        config = get_project_config()

    paths = config.paths
    feature_cfg = config.features

    logger.info("Starting feature engineering pipeline...")

    df_logs = pd.read_csv(paths.raw_logs_file, parse_dates=["timestamp"])
    df_truth = pd.read_csv(paths.ground_truth_file, parse_dates=["timestamp"])

    logger.info("Loaded raw_logs: %d rows, ground_truth: %d rows", len(df_logs), len(df_truth))

    df_logs = df_logs.sort_values("timestamp").reset_index(drop=True)
    df_truth = df_truth.sort_values("timestamp").reset_index(drop=True)

    if df_logs["timestamp"].dt.tz is None:
        df_logs["timestamp"] = df_logs["timestamp"].dt.tz_localize("UTC")
    if df_truth["timestamp"].dt.tz is None:
        df_truth["timestamp"] = df_truth["timestamp"].dt.tz_localize("UTC")

    _init_geo_coords(config)

    logger.info("Computing behavioural features...")
    behavioral_df = _compute_behavioral_features(df_logs)

    logger.info("Computing network features...")
    network_df = _compute_network_features(df_logs, feature_cfg)

    logger.info("Computing device & resource features...")
    device_resource_df = _compute_device_resource_features(df_logs, feature_cfg)

    logger.info("Computing relationship features...")
    relationship_df = _compute_relationship_features(df_logs)

    entity_type_onehot, encoder = _encode_entity_type(df_logs)

    X, feature_names = _assemble_features(
        entity_type_onehot,
        behavioral_df,
        network_df,
        device_resource_df,
        relationship_df,
    )

    y = df_truth["is_attack"].values.astype(np.float64)

    X_train, X_val, X_test, y_train, y_val, y_test = _temporal_split(
        X, y, feature_cfg
    )

    X_train, X_val, X_test, scaler = _fit_and_transform_scaler(
        X_train, X_val, X_test, feature_cfg.scaler_type
    )

    _save_artefacts(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names, scaler, encoder,
        paths, feature_cfg,
    )

    logger.info("Feature pipeline completed successfully.")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "encoder": encoder,
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    run_feature_pipeline()
