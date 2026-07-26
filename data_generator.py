"""Synthetic security log generator simulating realistic multi-persona behavioral events and injected attack patterns.

Simulates normal baseline activities and 7 MITRE attack types across 30 days, creating raw_logs.csv and ground_truth.csv.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from config import (
    ATTACK_TYPES,
    ENTITY_TYPES,
    GROUND_TRUTH_COLUMNS,
    RAW_LOG_COLUMNS,
    DataGenConfig,
    PathConfig,
    ProjectConfig,
    get_project_config,
)

logger = logging.getLogger(__name__)

_GEO_COORDS: Dict[str, Tuple[float, float]] = {}

def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float,
    earth_radius_km: float = 6371.0,
) -> float:
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_km * c

@dataclass
class EntityProfile:

    entity_id: str
    entity_type: str
    home_geo: Tuple[float, float, str]
    primary_fingerprint: str
    primary_auth: str
    auth_methods: Tuple[str, ...]
    auth_weights: np.ndarray
    primary_resources: Tuple[str, ...]
    command_vocab: Tuple[str, ...]
    ip_subnets: Tuple[str, ...]
    ip_subnet_weights: np.ndarray
    primary_ip: str = ""
    last_login_ts: Optional[datetime] = None
    login_count: int = 0
    ip_history: List[str] = field(default_factory=list)

def _set_deterministic_seeds(cfg: DataGenConfig) -> None:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.numpy_seed)
    logger.info(
        "Seeds set — random: %d, numpy: %d, faker: %d",
        cfg.random_seed,
        cfg.numpy_seed,
        cfg.faker_seed,
    )

def _create_faker(cfg: DataGenConfig) -> Faker:
    fake = Faker()
    Faker.seed(cfg.faker_seed)
    return fake

def _normalise_weights(weights: Tuple[float, ...]) -> np.ndarray:
    arr = np.array(weights, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total

def _generate_persona_ip(
    subnets: Tuple[str, ...],
    subnet_weights: np.ndarray,
    rng: np.random.Generator,
) -> str:
    idx = int(rng.choice(len(subnets), p=subnet_weights))
    template = subnets[idx]
    placeholders = template.count("{}")
    parts = [int(rng.integers(1, 255)) for _ in range(placeholders)]
    return template.format(*parts)

def _generate_command_sequence(
    vocab: Tuple[str, ...],
    rng: np.random.Generator,
    length_range: Tuple[int, int],
) -> str:
    seq_len = int(rng.integers(length_range[0], length_range[1] + 1))
    indices = rng.integers(0, len(vocab), size=seq_len)
    return ";".join(vocab[i] for i in indices)

def _sample_session_duration(
    entity_type: str,
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> float:
    ranges = {
        "corporate_employee": cfg.corporate_session_duration_range,
        "factory_operator": cfg.factory_session_duration_range,
        "service_account": cfg.service_session_duration_range,
    }
    lo, hi = ranges[entity_type]
    mu = (math.log(lo) + math.log(hi)) / 2.0
    sigma = (math.log(hi) - math.log(lo)) / 4.0
    duration = float(rng.lognormal(mu, sigma))
    return round(max(lo, min(hi, duration)), 2)

def _sample_timestamp_for_persona(
    entity_type: str,
    base_date: datetime,
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> datetime:
    if entity_type == "corporate_employee":
        if rng.random() < 0.90:
            hour = int(
                rng.integers(cfg.corporate_work_hour_start, cfg.corporate_work_hour_end)
            )
        else:
            hour = int(rng.integers(0, 24))
    elif entity_type == "factory_operator":
        shift_start = int(rng.choice(cfg.factory_shift_hours))
        hour = (shift_start + int(rng.integers(0, 8))) % 24
    else:
        hour = int(rng.integers(0, 24))

    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return base_date.replace(
        hour=hour, minute=minute, second=second, tzinfo=timezone.utc
    )

def _sample_persona_auth(
    profile: EntityProfile,
    rng: np.random.Generator,
) -> str:
    if rng.random() < 0.80:
        return profile.primary_auth
    idx = int(rng.choice(len(profile.auth_methods), p=profile.auth_weights))
    return profile.auth_methods[idx]

def _create_entity_profiles(
    cfg: DataGenConfig,
    fake: Faker,
    rng: np.random.Generator,
) -> List[EntityProfile]:
    profiles: List[EntityProfile] = []

    type_counts = np.round(
        np.array(cfg.entity_type_distribution) * cfg.num_entities
    ).astype(int)
    type_counts[-1] = cfg.num_entities - type_counts[:-1].sum()

    entity_idx = 0
    for etype_idx, etype in enumerate(ENTITY_TYPES):
        count = int(type_counts[etype_idx])

        if etype == "corporate_employee":
            resources = cfg.corporate_resources
            commands = cfg.corporate_commands
            fingerprints = cfg.corporate_fingerprints
            auth_meths = cfg.auth_methods
            auth_wts = _normalise_weights(cfg.corporate_auth_weights)
            ip_subs = cfg.corporate_ip_subnets
            ip_wts = _normalise_weights(cfg.corporate_ip_subnet_weights)
        elif etype == "factory_operator":
            resources = cfg.factory_resources
            commands = cfg.factory_commands
            fingerprints = cfg.factory_fingerprints
            auth_meths = cfg.auth_methods
            auth_wts = _normalise_weights(cfg.factory_auth_weights)
            ip_subs = cfg.factory_ip_subnets
            ip_wts = _normalise_weights(cfg.factory_ip_subnet_weights)
        else:
            resources = cfg.service_resources
            commands = cfg.service_commands
            fingerprints = cfg.service_fingerprints
            auth_meths = cfg.auth_methods + cfg.service_extra_auth_methods
            base_wts = cfg.service_auth_weights + cfg.service_extra_auth_weights
            auth_wts = _normalise_weights(base_wts)
            ip_subs = cfg.service_ip_subnets
            ip_wts = _normalise_weights(cfg.service_ip_subnet_weights)

        for _ in range(count):
            entity_id = f"{etype[:3].upper()}-{entity_idx:05d}"
            home_geo = cfg.geo_locations[int(rng.integers(0, len(cfg.geo_locations)))]
            primary_fp = fingerprints[int(rng.integers(0, len(fingerprints)))]

            primary_auth_idx = int(rng.choice(len(auth_meths), p=auth_wts))
            primary_auth = auth_meths[primary_auth_idx]

            num_resources = max(2, int(rng.integers(2, len(resources) + 1)))
            resource_indices = rng.choice(
                len(resources), size=num_resources, replace=False
            )
            entity_resources = tuple(resources[i] for i in resource_indices)

            primary_ip = _generate_persona_ip(ip_subs, ip_wts, rng)

            profiles.append(
                EntityProfile(
                    entity_id=entity_id,
                    entity_type=etype,
                    home_geo=home_geo,
                    primary_fingerprint=primary_fp,
                    primary_auth=primary_auth,
                    auth_methods=auth_meths,
                    auth_weights=auth_wts,
                    primary_resources=entity_resources,
                    command_vocab=commands,
                    ip_subnets=ip_subs,
                    ip_subnet_weights=ip_wts,
                    primary_ip=primary_ip,
                )
            )
            entity_idx += 1

    logger.info(
        "Created %d entity profiles: %s",
        len(profiles),
        {etype: int(c) for etype, c in zip(ENTITY_TYPES, type_counts)},
    )
    return profiles

def _generate_normal_event(
    profile: EntityProfile,
    timestamp: datetime,
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if profile.ip_history and rng.random() < 0.70:
        source_ip = profile.ip_history[int(rng.integers(0, len(profile.ip_history)))]
    elif rng.random() < 0.50:
        source_ip = profile.primary_ip
    else:
        source_ip = _generate_persona_ip(
            profile.ip_subnets, profile.ip_subnet_weights, rng
        )
    if source_ip not in profile.ip_history:
        profile.ip_history.append(source_ip)
        if len(profile.ip_history) > 5:
            profile.ip_history = profile.ip_history[-5:]

    if rng.random() < 0.85:
        geo_label = profile.home_geo[2]
    else:
        nearby = cfg.geo_locations[int(rng.integers(0, len(cfg.geo_locations)))]
        geo_label = nearby[2]

    resource = profile.primary_resources[
        int(rng.integers(0, len(profile.primary_resources)))
    ]

    auth = _sample_persona_auth(profile, rng)

    session_dur = _sample_session_duration(profile.entity_type, cfg, rng)

    cmd_seq = _generate_command_sequence(
        profile.command_vocab, rng, cfg.command_sequence_length_range
    )

    if rng.random() < 0.90:
        fingerprint = profile.primary_fingerprint
    else:
        if profile.entity_type == "corporate_employee":
            fps = cfg.corporate_fingerprints
        elif profile.entity_type == "factory_operator":
            fps = cfg.factory_fingerprints
        else:
            fps = cfg.service_fingerprints
        fingerprint = fps[int(rng.integers(0, len(fps)))]

    profile.last_login_ts = timestamp
    profile.login_count += 1

    return {
        "entity_id": profile.entity_id,
        "entity_type": profile.entity_type,
        "timestamp": timestamp.isoformat(),
        "source_ip": source_ip,
        "geo_location": geo_label,
        "resource_accessed": resource,
        "auth_method": auth,
        "session_duration": session_dur,
        "command_sequence": cmd_seq,
        "device_fingerprint": fingerprint,
    }

def _inject_brute_force(
    event: Dict[str, Any],
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    event["auth_method"] = "password"
    event["session_duration"] = round(float(rng.uniform(0.01, 0.5)), 2)
    event["command_sequence"] = ";".join(["login_attempt"] * int(rng.integers(5, 20)))
    if rng.random() < 0.40:
        event["source_ip"] = f"172.16.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
    return event

def _inject_credential_stuffing(
    event: Dict[str, Any],
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    event["source_ip"] = f"172.{rng.integers(16, 32)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
    event["auth_method"] = "password"
    event["session_duration"] = round(float(rng.uniform(0.01, 1.0)), 2)
    event["command_sequence"] = ";".join(
        ["auth_check", "login_attempt"] * int(rng.integers(2, 6))
    )
    return event

def _inject_impossible_travel(
    event: Dict[str, Any],
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    current_label = event["geo_location"]
    current_coords = _GEO_COORDS.get(current_label)

    min_dist = cfg.impossible_travel_min_distance_km
    far_candidates = []
    for loc in cfg.geo_locations:
        if loc[2] == current_label:
            continue
        if current_coords is not None:
            dist = _haversine_km(
                current_coords[0], current_coords[1], loc[0], loc[1]
            )
            if dist >= min_dist:
                far_candidates.append(loc)
        else:
            far_candidates.append(loc)

    if not far_candidates:
        far_candidates = [loc for loc in cfg.geo_locations if loc[2] != current_label]

    if far_candidates:
        far_loc = far_candidates[int(rng.integers(0, len(far_candidates)))]
        event["geo_location"] = far_loc[2]

    ts = datetime.fromisoformat(event["timestamp"])
    gap_seconds = int(rng.integers(60, cfg.impossible_travel_max_gap_seconds + 1))
    ts = ts - timedelta(seconds=gap_seconds)
    event["timestamp"] = ts.isoformat()

    event["source_ip"] = f"172.16.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
    return event

def _inject_lateral_movement(
    event: Dict[str, Any],
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    entity_type = event["entity_type"]
    if entity_type == "corporate_employee":
        cross_resources = cfg.factory_resources + cfg.service_resources
    elif entity_type == "factory_operator":
        cross_resources = cfg.corporate_resources + cfg.service_resources
    else:
        cross_resources = cfg.corporate_resources + cfg.factory_resources

    event["resource_accessed"] = cross_resources[
        int(rng.integers(0, len(cross_resources)))
    ]
    lateral_cmds = [
        "net_scan", "port_scan", "whoami", "net_user", "tasklist",
        "reg_query", "psexec", "wmic", "mimikatz", "pass_the_hash",
    ]
    num_cmds = int(rng.integers(3, 8))
    event["command_sequence"] = ";".join(
        lateral_cmds[int(rng.integers(0, len(lateral_cmds)))]
        for _ in range(num_cmds)
    )
    return event

def _inject_device_spoofing(
    event: Dict[str, Any],
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    os_choice = cfg.spoofed_os_options[
        int(rng.integers(0, len(cfg.spoofed_os_options)))
    ]
    browser_choice = cfg.spoofed_browser_options[
        int(rng.integers(0, len(cfg.spoofed_browser_options)))
    ]
    arch_choice = cfg.spoofed_arch_options[
        int(rng.integers(0, len(cfg.spoofed_arch_options)))
    ]
    profile_choice = cfg.spoofed_profile_options[
        int(rng.integers(0, len(cfg.spoofed_profile_options)))
    ]
    event["device_fingerprint"] = f"{os_choice}-{browser_choice}-{arch_choice}-{profile_choice}"

    if rng.random() < 0.50:
        event["source_ip"] = f"172.16.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
    return event

def _inject_low_and_slow_exfiltration(
    event: Dict[str, Any],
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    event["session_duration"] = round(float(rng.uniform(300.0, 720.0)), 2)
    exfil_cmds = [
        "scp", "rsync", "curl_upload", "compress", "encrypt",
        "split_file", "base64_encode", "dns_tunnel",
    ]
    num_cmds = int(rng.integers(4, 10))
    event["command_sequence"] = ";".join(
        exfil_cmds[int(rng.integers(0, len(exfil_cmds)))]
        for _ in range(num_cmds)
    )
    sensitive_resources = ["finance_dashboard", "hr_portal", "historian_db", "crm_system"]
    event["resource_accessed"] = sensitive_resources[
        int(rng.integers(0, len(sensitive_resources)))
    ]
    return event

def _inject_insider_drift(
    event: Dict[str, Any],
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    ts = datetime.fromisoformat(event["timestamp"])
    unusual_hour = int(rng.choice([0, 1, 2, 3, 4, 5, 23, 22, 21]))
    ts = ts.replace(hour=unusual_hour)
    event["timestamp"] = ts.isoformat()

    entity_type = event["entity_type"]
    if entity_type == "corporate_employee":
        cross_resources = cfg.factory_resources + cfg.service_resources
    elif entity_type == "factory_operator":
        cross_resources = cfg.corporate_resources + cfg.service_resources
    else:
        cross_resources = cfg.corporate_resources + cfg.factory_resources

    event["resource_accessed"] = cross_resources[
        int(rng.integers(0, len(cross_resources)))
    ]

    event["auth_method"] = cfg.auth_methods[
        int(rng.integers(0, len(cfg.auth_methods)))
    ]

    return event

_ATTACK_INJECTORS = {
    "brute_force": _inject_brute_force,
    "credential_stuffing": _inject_credential_stuffing,
    "impossible_travel": _inject_impossible_travel,
    "lateral_movement": _inject_lateral_movement,
    "device_spoofing": _inject_device_spoofing,
    "low_and_slow_exfiltration": _inject_low_and_slow_exfiltration,
    "insider_drift": _inject_insider_drift,
}

def _generate_all_events(
    profiles: List[EntityProfile],
    cfg: DataGenConfig,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    attack_rate = float(
        rng.uniform(cfg.attack_rate_min, cfg.attack_rate_max)
    )
    num_attacks = int(cfg.num_events * attack_rate)
    attack_indices = set(
        rng.choice(cfg.num_events, size=num_attacks, replace=False).tolist()
    )
    logger.info(
        "Attack budget: %d events (%.2f%% of %d)",
        num_attacks,
        attack_rate * 100,
        cfg.num_events,
    )

    attack_weights = _normalise_weights(cfg.attack_type_weights)

    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    day_offsets = rng.integers(0, cfg.simulation_days, size=cfg.num_events)

    for event_idx in range(cfg.num_events):
        profile = profiles[int(rng.integers(0, len(profiles)))]

        base_date = start_date + timedelta(days=int(day_offsets[event_idx]))

        timestamp = _sample_timestamp_for_persona(
            profile.entity_type, base_date, cfg, rng
        )

        event = _generate_normal_event(profile, timestamp, cfg, rng)

        is_attack = event_idx in attack_indices
        attack_type = "none"

        if is_attack:
            attack_type_idx = int(rng.choice(len(ATTACK_TYPES), p=attack_weights))
            attack_type = ATTACK_TYPES[attack_type_idx]
            injector = _ATTACK_INJECTORS[attack_type]
            event = injector(event, cfg, rng)

        events.append(event)
        ground_truth.append(
            {
                "entity_id": event["entity_id"],
                "timestamp": event["timestamp"],
                "is_attack": int(is_attack),
                "attack_type": attack_type,
            }
        )

        if (event_idx + 1) % 25_000 == 0:
            logger.info("Generated %d / %d events", event_idx + 1, cfg.num_events)

    return events, ground_truth

def _validate_dataset(
    df_logs: pd.DataFrame,
    df_truth: pd.DataFrame,
    cfg: DataGenConfig,
) -> None:
    errors: List[str] = []

    missing_log_cols = set(RAW_LOG_COLUMNS) - set(df_logs.columns)
    if missing_log_cols:
        errors.append(f"Missing columns in raw_logs: {missing_log_cols}")

    missing_gt_cols = set(GROUND_TRUTH_COLUMNS) - set(df_truth.columns)
    if missing_gt_cols:
        errors.append(f"Missing columns in ground_truth: {missing_gt_cols}")

    if len(df_logs) != len(df_truth):
        errors.append(
            f"Row count mismatch: logs={len(df_logs)}, truth={len(df_truth)}"
        )

    null_counts = df_logs.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        errors.append(f"Null values detected in raw_logs:\n{cols_with_nulls}")

    attack_rate = df_truth["is_attack"].mean()
    tolerance = 0.005
    if attack_rate < (cfg.attack_rate_min - tolerance) or attack_rate > (
        cfg.attack_rate_max + tolerance
    ):
        errors.append(
            f"Attack rate {attack_rate:.4f} outside bounds "
            f"[{cfg.attack_rate_min}, {cfg.attack_rate_max}]"
        )

    valid_types = set(ENTITY_TYPES)
    invalid_types = set(df_logs["entity_type"].unique()) - valid_types
    if invalid_types:
        errors.append(f"Invalid entity types found: {invalid_types}")

    for etype in ENTITY_TYPES:
        mask = df_logs["entity_type"] == etype
        if mask.sum() == 0:
            continue
        auth_dist = df_logs.loc[mask, "auth_method"].value_counts(normalize=True)
        if etype == "service_account":
            biometric_rate = auth_dist.get("biometric", 0.0)
            if biometric_rate > 0.01:
                errors.append(
                    f"Service accounts have biometric auth rate={biometric_rate:.3f} "
                    f"(expected ~0)"
                )
        if etype == "corporate_employee":
            mfa_sso_rate = auth_dist.get("mfa", 0.0) + auth_dist.get("sso", 0.0)
            if mfa_sso_rate < 0.10:
                errors.append(
                    f"Corporate employees have low MFA+SSO rate={mfa_sso_rate:.3f}"
                )

    logger.info("Auth distribution validation complete.")

    corporate_res_set = set(cfg.corporate_resources)
    factory_res_set = set(cfg.factory_resources)
    service_res_set = set(cfg.service_resources)

    for etype, expected_res in [
        ("corporate_employee", corporate_res_set),
        ("factory_operator", factory_res_set),
        ("service_account", service_res_set),
    ]:
        mask = df_logs["entity_type"] == etype
        if mask.sum() == 0:
            continue
        non_attack_mask = mask & (df_truth["is_attack"] == 0)
        if non_attack_mask.sum() == 0:
            continue
        resources_used = set(df_logs.loc[non_attack_mask, "resource_accessed"].unique())
        cross_domain = resources_used - expected_res
        cross_domain_rate = (
            df_logs.loc[non_attack_mask, "resource_accessed"]
            .isin(cross_domain)
            .mean()
        )
        if cross_domain_rate > 0.05:
            errors.append(
                f"{etype} has {cross_domain_rate:.3f} cross-domain resource "
                f"access rate in normal events (expected <0.05)"
            )

    logger.info("Resource distribution validation complete.")

    attack_events = df_truth[df_truth["is_attack"] == 1]
    if len(attack_events) > 0:
        attack_dist = (
            attack_events["attack_type"]
            .value_counts(normalize=True)
            .to_dict()
        )
        expected_weights = _normalise_weights(cfg.attack_type_weights)
        for i, atype in enumerate(ATTACK_TYPES):
            actual = attack_dist.get(atype, 0.0)
            expected = float(expected_weights[i])
            if expected > 0.05 and actual < expected * 0.20:
                errors.append(
                    f"Attack type '{atype}' has rate={actual:.3f}, "
                    f"expected ~{expected:.3f} (too low)"
                )

    logger.info("Attack distribution validation complete.")

    for etype, expected_prefixes in [
        ("corporate_employee", ("10.1.", "10.200.", "192.168.1.")),
        ("factory_operator", ("10.10.", "10.20.")),
        ("service_account", ("10.100.0.", "10.100.1.")),
    ]:
        mask = df_logs["entity_type"] == etype
        non_attack_mask = mask & (df_truth["is_attack"] == 0)
        if non_attack_mask.sum() == 0:
            continue
        ips = df_logs.loc[non_attack_mask, "source_ip"]
        in_range = ips.apply(
            lambda ip: any(ip.startswith(p) for p in expected_prefixes)
        )
        in_range_rate = in_range.mean()
        if in_range_rate < 0.80:
            errors.append(
                f"{etype} has {in_range_rate:.3f} in-subnet IP rate "
                f"(expected >=0.80, prefixes={expected_prefixes})"
            )

    logger.info("IP range validation complete.")

    valid_geos = {loc[2] for loc in cfg.geo_locations}
    invalid_geos = set(df_logs["geo_location"].unique()) - valid_geos
    if invalid_geos:
        errors.append(f"Invalid geo_locations found: {invalid_geos}")

    all_valid_fps = (
        set(cfg.corporate_fingerprints)
        | set(cfg.factory_fingerprints)
        | set(cfg.service_fingerprints)
    )
    non_attack_mask = df_truth["is_attack"] == 0
    normal_fps = set(df_logs.loc[non_attack_mask, "device_fingerprint"].unique())
    invalid_fps = normal_fps - all_valid_fps
    if invalid_fps:
        errors.append(
            f"Non-attack events contain invalid fingerprints: {invalid_fps}"
        )

    if errors:
        error_msg = "Dataset validation FAILED:\n" + "\n".join(
            f"  [{i+1}] {e}" for i, e in enumerate(errors)
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        "Validation PASSED — %d events, attack rate: %.4f, %d checks OK",
        len(df_logs),
        attack_rate,
        12,
    )

def generate_dataset(config: Optional[ProjectConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = get_project_config()

    cfg = config.data_gen
    paths = config.paths

    logger.info("Starting synthetic data generation...")
    _set_deterministic_seeds(cfg)
    rng = np.random.default_rng(cfg.numpy_seed)
    fake = _create_faker(cfg)

    global _GEO_COORDS
    _GEO_COORDS = {
        label: (lat, lon)
        for lat, lon, label in cfg.geo_locations
    }

    profiles = _create_entity_profiles(cfg, fake, rng)

    events, ground_truth = _generate_all_events(profiles, cfg, rng)

    test_window_start = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=28)
    demo_events = []
    demo_truth = []
    current_time = test_window_start
    
    for i in range(20):
        is_attack = 1 if i == 19 else 0
        attack_type = "impossible_travel" if is_attack else "none"
        
        event = {
            "timestamp": current_time.isoformat(),
            "entity_id": "cold_start_demo_user",
            "entity_type": "corporate_employee",
            "source_ip": "10.0.0.99" if not is_attack else "10.0.99.99",
            "geo_location": "New York" if not is_attack else "Moscow",
            "auth_method": "mfa" if not is_attack else "password",
            "action": "login",
            "status": "success",
            "device_fingerprint": cfg.corporate_fingerprints[0] if not is_attack else "Windows 95",
            "resource_accessed": cfg.corporate_resources[0],
            "command_sequence": "whoami",
            "session_duration": 60.0
        }
        truth = {
            "timestamp": current_time.isoformat(),
            "entity_id": "cold_start_demo_user",
            "is_attack": is_attack,
            "attack_type": attack_type
        }
        events.append(event)
        ground_truth.append(truth)
        current_time += timedelta(minutes=15)

    chain_entity_id = "chain_demo_user"
    chain_start = test_window_start + timedelta(days=8)
    baseline_event = {
        "entity_id": chain_entity_id,
        "entity_type": "corporate_employee",
        "source_ip": "10.0.0.77",
        "geo_location": "New York",
        "resource_accessed": cfg.corporate_resources[0],
        "auth_method": "mfa",
        "session_duration": 60.0,
        "command_sequence": "outlook;teams;whoami",
        "device_fingerprint": cfg.corporate_fingerprints[0],
    }
    for i in range(8):
        timestamp = chain_start + timedelta(days=i)
        event = {**baseline_event, "timestamp": timestamp.isoformat()}
        events.append(event)
        ground_truth.append({
            "timestamp": timestamp.isoformat(),
            "entity_id": chain_entity_id,
            "is_attack": 0,
            "attack_type": "none",
        })

    stage_start = chain_start + timedelta(days=7, hours=3)
    credential_attempt_fields = {
        "source_ip": "10.0.0.77",
        "geo_location": "New York",
        "resource_accessed": "vpn_gateway",
        "auth_method": "password",
        "session_duration": 0.1,
        "command_sequence": "auth_check;login_attempt",
        "device_fingerprint": cfg.corporate_fingerprints[0],
    }
    for attempt_index in range(5):
        timestamp = stage_start - timedelta(minutes=5 - attempt_index)
        events.append({
            "entity_id": chain_entity_id,
            "entity_type": "corporate_employee",
            "timestamp": timestamp.isoformat(),
            **credential_attempt_fields,
        })
        ground_truth.append({
            "timestamp": timestamp.isoformat(),
            "entity_id": chain_entity_id,
            "is_attack": 1,
            "attack_type": "credential_stuffing",
        })

    chain_stages = [
        (
            "credential_stuffing",
            {
                "source_ip": "10.0.0.77",
                "geo_location": "New York",
                "resource_accessed": "vpn_gateway",
                "auth_method": "password",
                "session_duration": 0.1,
                "command_sequence": "auth_check;login_attempt;auth_check;login_attempt;auth_check;login_attempt",
                "device_fingerprint": cfg.corporate_fingerprints[0],
            },
        ),
        (
            "lateral_movement",
            {
                "source_ip": "10.0.0.77",
                "geo_location": "New York",
                "resource_accessed": "plc_controller",
                "auth_method": "password",
                "session_duration": 180.0,
                "command_sequence": "net_scan;port_scan;psexec;mimikatz;pass_the_hash;net_user",
                "device_fingerprint": cfg.corporate_fingerprints[0],
            },
        ),
        (
            "device_spoofing",
            {
                "source_ip": "10.0.0.77",
                "geo_location": "New York",
                "resource_accessed": "certificate_authority",
                "auth_method": "password",
                "session_duration": 240.0,
                "command_sequence": "sudo;runas;chmod;reg_query;wmic;whoami",
                "device_fingerprint": "Kali-Firefox-x64",
            },
        ),
        (
            "low_and_slow_exfiltration",
            {
                "source_ip": "10.0.0.77",
                "geo_location": "New York",
                "resource_accessed": "finance_dashboard",
                "auth_method": "password",
                "session_duration": 1440.0,
                "command_sequence": "compress;encrypt;split_file;scp;curl_upload;dns_tunnel;rsync;base64_encode;archive;stage_data",
                "device_fingerprint": cfg.corporate_fingerprints[0],
            },
        ),
    ]
    for stage_index, (attack_type, stage_fields) in enumerate(chain_stages):
        timestamp = stage_start + timedelta(minutes=10 * stage_index)
        event = {
            "entity_id": chain_entity_id,
            "entity_type": "corporate_employee",
            "timestamp": timestamp.isoformat(),
            **stage_fields,
        }
        events.append(event)
        ground_truth.append({
            "timestamp": timestamp.isoformat(),
            "entity_id": chain_entity_id,
            "is_attack": 1,
            "attack_type": attack_type,
        })

    df_logs = pd.DataFrame(events, columns=list(RAW_LOG_COLUMNS))
    df_truth = pd.DataFrame(ground_truth, columns=list(GROUND_TRUTH_COLUMNS))

    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], utc=True)
    df_truth["timestamp"] = pd.to_datetime(df_truth["timestamp"], utc=True)

    sort_idx = df_logs["timestamp"].argsort()
    df_logs = df_logs.iloc[sort_idx].reset_index(drop=True)
    df_truth = df_truth.iloc[sort_idx].reset_index(drop=True)

    _validate_dataset(df_logs, df_truth, cfg)

    paths.ensure_dirs()
    df_logs.to_csv(paths.raw_logs_file, index=False)
    df_truth.to_csv(paths.ground_truth_file, index=False)

    logger.info("Wrote raw_logs.csv (%d rows) to %s", len(df_logs), paths.raw_logs_file)
    logger.info(
        "Wrote ground_truth.csv (%d rows) to %s",
        len(df_truth),
        paths.ground_truth_file,
    )

    attack_counts = df_truth[df_truth["is_attack"] == 1]["attack_type"].value_counts()
    logger.info("Attack distribution:\n%s", attack_counts.to_string())
    entity_type_counts = df_logs["entity_type"].value_counts()
    logger.info("Entity type distribution:\n%s", entity_type_counts.to_string())

    return df_logs, df_truth

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    generate_dataset()
