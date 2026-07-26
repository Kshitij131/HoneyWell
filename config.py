"""Central project configuration, path management, and system hyperparameters.

Defines all filesystem paths, persona types, attack classification labels,
feature definitions, and neural network training parameters.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ===========================================================================
# Project Path Configuration
# ===========================================================================
@dataclass(frozen=True)
class PathConfig:

    project_root: Path = _PROJECT_ROOT
    raw_data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed"
    )
    models_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "models")
    outputs_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "outputs")

    raw_logs_file: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "raw" / "raw_logs.csv"
    )
    ground_truth_file: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "raw" / "ground_truth.csv"
    )
    risk_weights_file: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "risk_weights.json"
    )
    risk_thresholds_file: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "risk_thresholds.json"
    )
    platt_scaler_file: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "platt_scaler.pkl"
    )

    def ensure_dirs(self) -> None:
        for dir_path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.outputs_dir,
        ):
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", dir_path)

RAW_LOG_COLUMNS: Tuple[str, ...] = (
    "entity_id",
    "entity_type",
    "timestamp",
    "source_ip",
    "geo_location",
    "resource_accessed",
    "auth_method",
    "session_duration",
    "command_sequence",
    "device_fingerprint",
)

GROUND_TRUTH_COLUMNS: Tuple[str, ...] = (
    "entity_id",
    "timestamp",
    "is_attack",
    "attack_type",
)

ENTITY_TYPES: Tuple[str, ...] = (
    "corporate_employee",
    "factory_operator",
    "service_account",
)

ENTITY_TYPE_TO_INDEX: Dict[str, int] = {
    etype: idx for idx, etype in enumerate(ENTITY_TYPES)
}

NUM_ENTITY_TYPES: int = len(ENTITY_TYPES)

ATTACK_TYPES: Tuple[str, ...] = (
    "brute_force",
    "credential_stuffing",
    "impossible_travel",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfiltration",
    "insider_drift",
)

NUM_ATTACK_TYPES: int = len(ATTACK_TYPES)

ATTACK_TYPE_LABELS: Dict[str, int] = {
    "benign": 0,
    **{atype: idx + 1 for idx, atype in enumerate(ATTACK_TYPES)},
}
ATTACK_TYPE_LABELS_INV: Dict[int, str] = {v: k for k, v in ATTACK_TYPE_LABELS.items()}
NUM_ATTACK_CLASSES: int = len(ATTACK_TYPE_LABELS)

@dataclass(frozen=True)
class DataGenConfig:

    num_entities: int = 500
    num_events: int = 100_000
    entity_type_distribution: Tuple[float, ...] = (0.50, 0.30, 0.20)
    attack_rate_min: float = 0.005
    attack_rate_max: float = 0.03
    simulation_days: int = 30
    random_seed: int = 42
    faker_seed: int = 42
    numpy_seed: int = 42

    geo_locations: Tuple[Tuple[float, float, str], ...] = (
        (37.7749, -122.4194, "San Francisco"),
        (40.7128, -74.0060, "New York"),
        (51.5074, -0.1278, "London"),
        (35.6762, 139.6503, "Tokyo"),
        (1.3521, 103.8198, "Singapore"),
        (48.8566, 2.3522, "Paris"),
        (55.7558, 37.6173, "Moscow"),
        (28.6139, 77.2090, "New Delhi"),
        (-33.8688, 151.2093, "Sydney"),
        (52.5200, 13.4050, "Berlin"),
    )

    auth_methods: Tuple[str, ...] = (
        "password",
        "mfa",
        "sso",
        "certificate",
        "api_key",
        "biometric",
    )

    corporate_auth_weights: Tuple[float, ...] = (
        0.25,
        0.25,
        0.25,
        0.03,
        0.02,
        0.20,
    )
    factory_auth_weights: Tuple[float, ...] = (
        0.35,
        0.25,
        0.05,
        0.20,
        0.05,
        0.10,
    )
    service_auth_weights: Tuple[float, ...] = (
        0.02,
        0.00,
        0.05,
        0.40,
        0.40,
        0.00,
    )
    service_extra_auth_methods: Tuple[str, ...] = ("token",)
    service_extra_auth_weights: Tuple[float, ...] = (0.13,)

    corporate_resources: Tuple[str, ...] = (
        "email_server",
        "crm_system",
        "hr_portal",
        "sharepoint",
        "vpn_gateway",
        "code_repository",
        "internal_wiki",
        "finance_dashboard",
    )
    factory_resources: Tuple[str, ...] = (
        "plc_controller",
        "scada_hmi",
        "historian_db",
        "safety_system",
        "robot_arm_interface",
        "sensor_gateway",
        "batch_scheduler",
        "maintenance_portal",
    )
    service_resources: Tuple[str, ...] = (
        "telemetry_endpoint",
        "config_server",
        "firmware_update_api",
        "certificate_authority",
        "ntp_server",
        "log_aggregator",
        "health_check_api",
        "ota_update_server",
    )

    corporate_commands: Tuple[str, ...] = (
        "ls", "cd", "cat", "grep", "git", "ssh", "scp", "curl",
        "docker", "kubectl", "python", "pip", "code", "outlook",
        "teams", "slack", "whoami", "chmod", "nano", "vim",
    )
    factory_commands: Tuple[str, ...] = (
        "read_register", "write_register", "set_point", "get_status",
        "start_batch", "stop_batch", "calibrate", "acknowledge_alarm",
        "download_firmware", "upload_config", "restart_plc",
        "read_sensor", "set_threshold", "toggle_valve", "run_diagnostic",
    )
    service_commands: Tuple[str, ...] = (
        "heartbeat", "send_telemetry", "fetch_config", "check_update",
        "rotate_cert", "sync_time", "report_status", "upload_logs",
        "ping", "ack", "register", "deregister", "self_test",
    )

    corporate_fingerprints: Tuple[str, ...] = (
        "Win11-Chrome-x64",
        "Win10-Edge-x64",
        "macOS-Safari-arm64",
        "macOS-Chrome-arm64",
        "Ubuntu-Firefox-x64",
    )
    factory_fingerprints: Tuple[str, ...] = (
        "WinCE-HMI-Panel",
        "Linux-SCADA-x86",
        "RTOS-PLC-arm",
        "VxWorks-DCS-mips",
    )
    service_fingerprints: Tuple[str, ...] = (
        "Linux-IoT-arm",
        "FreeRTOS-Sensor-cortex",
        "Zephyr-Gateway-riscv",
        "Embedded-Linux-mips",
    )

    corporate_work_hour_start: int = 8
    corporate_work_hour_end: int = 18
    factory_shift_hours: Tuple[int, ...] = (6, 14, 22)
    service_interval_minutes_mean: float = 5.0
    service_interval_minutes_std: float = 2.0

    corporate_session_duration_range: Tuple[float, float] = (5.0, 480.0)
    factory_session_duration_range: Tuple[float, float] = (10.0, 720.0)
    service_session_duration_range: Tuple[float, float] = (0.1, 5.0)

    command_sequence_length_range: Tuple[int, int] = (1, 8)

    corporate_ip_subnets: Tuple[str, ...] = (
        "10.1.{}.{}",
        "10.200.{}.{}",
        "192.168.1.{}",
    )
    corporate_ip_subnet_weights: Tuple[float, ...] = (0.55, 0.30, 0.15)
    factory_ip_subnets: Tuple[str, ...] = (
        "10.10.{}.{}",
        "10.20.{}.{}",
    )
    factory_ip_subnet_weights: Tuple[float, ...] = (0.60, 0.40)
    service_ip_subnets: Tuple[str, ...] = (
        "10.100.0.{}",
        "10.100.1.{}",
    )
    service_ip_subnet_weights: Tuple[float, ...] = (0.50, 0.50)

    attack_type_weights: Tuple[float, ...] = (
        0.35,
        0.25,
        0.15,
        0.08,
        0.10,
        0.05,
        0.02,
    )

    spoofed_os_options: Tuple[str, ...] = (
        "Win11", "Win10", "macOS", "Ubuntu", "Android", "Kali",
    )
    spoofed_browser_options: Tuple[str, ...] = (
        "Chrome", "Firefox", "Edge", "Headless-Chrome", "Selenium",
        "Puppeteer", "curl",
    )
    spoofed_arch_options: Tuple[str, ...] = (
        "x64", "x86", "arm64", "arm", "mips",
    )
    spoofed_profile_options: Tuple[str, ...] = (
        "Desktop", "Mobile", "VM", "Emulator", "Container",
    )

    impossible_travel_max_speed_kmh: float = 900.0
    impossible_travel_min_distance_km: float = 5000.0
    impossible_travel_max_gap_seconds: int = 1800

@dataclass(frozen=True)
class FeatureConfig:

    behavioral_features: Tuple[str, ...] = (
        "login_hour",
        "day_of_week",
        "session_duration",
        "time_since_last_login",
    )
    network_features: Tuple[str, ...] = (
        "unique_ips_last_hour",
        "geo_distance_km",
        "failed_logins_last_10min",
        "travel_speed_kmh",
    )
    device_resource_features: Tuple[str, ...] = (
        "fingerprint_consistency",
        "resource_rarity",
        "sequence_entropy",
        "privilege_escalation_count",
        "os_mismatch",
        "browser_mismatch",
    )
    relationship_features: Tuple[str, ...] = (
        "degree_centrality",
        "newly_observed_neighbors",
    )

    unique_ip_window_seconds: int = 3600
    failed_login_window_seconds: int = 600

    geo_distance_earth_radius_km: float = 6371.0

    sequence_entropy_base: int = 2

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    scaler_type: str = "standard"

    x_train_file: str = "X_train.npy"
    x_val_file: str = "X_val.npy"
    x_test_file: str = "X_test.npy"
    y_train_file: str = "y_train.npy"
    y_val_file: str = "y_val.npy"
    y_test_file: str = "y_test.npy"
    feature_metadata_file: str = "feature_metadata.json"
    scaler_file: str = "scaler.pkl"
    encoder_file: str = "encoder.pkl"

    @property
    def all_feature_names(self) -> Tuple[str, ...]:
        return (
            *self.behavioral_features,
            *self.network_features,
            *self.device_resource_features,
            *self.relationship_features,
        )

    @property
    def num_raw_features(self) -> int:
        return len(self.all_feature_names)

@dataclass(frozen=True)
class AutoencoderConfig:

    encoder_dims: Tuple[int, ...] = (128, 64)
    latent_dim: int = 32
    decoder_dims: Tuple[int, ...] = (64, 128)
    dropout_rate: float = 0.2
    activation: str = "relu"
    threshold_percentile: float = 95.0

@dataclass(frozen=True)
class LSTMConfig:

    hidden_dim: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout_rate: float = 0.3
    sequence_length: int = 10
    fc_dims: Tuple[int, ...] = (64,)

@dataclass(frozen=True)
class FocalLossConfig:

    alpha: float = 0.85
    gamma: float = 2.0

@dataclass(frozen=True)
class AttackClassifierConfig:

    fc_dims: Tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.3
    num_classes: int = 8
    num_epochs: int = 20
    learning_rate: float = 1e-3
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0

@dataclass(frozen=True)
class RiskScorerConfig:

    w_ae: float = 0.4
    w_lstm: float = 0.4
    w_policy: float = 0.2

@dataclass(frozen=True)
class PolicyRule:

    name: str
    feature: str
    operator: str
    threshold: float
    score: float

@dataclass(frozen=True)
class PolicyEngineConfig:

    rules: Tuple[PolicyRule, ...] = (
        PolicyRule(
            name="impossible_travel_speed",
            feature="travel_speed_kmh",
            operator="gt",
            threshold=900.0,
            score=1.5,
        ),
        PolicyRule(
            name="excessive_failed_logins",
            feature="failed_logins_last_10min",
            operator="gt",
            threshold=5.0,
            score=0.25,
        ),
        PolicyRule(
            name="unusual_login_hour",
            feature="login_hour",
            operator="gt",
            threshold=22.0,
            score=0.15,
        ),
        PolicyRule(
            name="high_sequence_entropy",
            feature="sequence_entropy",
            operator="gt",
            threshold=3.0,
            score=0.15,
        ),
        PolicyRule(
            name="low_fingerprint_consistency",
            feature="fingerprint_consistency",
            operator="lt",
            threshold=0.5,
            score=0.2,
        ),
        PolicyRule(
            name="os_mismatch",
            feature="os_mismatch",
            operator="eq",
            threshold=1.0,
            score=1.5,
        ),
        PolicyRule(
            name="browser_mismatch",
            feature="browser_mismatch",
            operator="eq",
            threshold=1.0,
            score=1.5,
        ),
        PolicyRule(
            name="high_unique_ips",
            feature="unique_ips_last_hour",
            operator="gt",
            threshold=10.0,
            score=0.15,
        ),
        PolicyRule(
            name="privilege_escalation",
            feature="privilege_escalation_count",
            operator="gt",
            threshold=2.0,
            score=0.25,
        ),
        PolicyRule(
            name="resource_rarity_spike",
            feature="resource_rarity",
            operator="gt",
            threshold=0.9,
            score=0.15,
        ),
    )
    max_score: float = 5.8

@dataclass(frozen=True)
class ModelConfig:

    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    focal_loss: FocalLossConfig = field(default_factory=FocalLossConfig)
    risk_scorer: RiskScorerConfig = field(default_factory=RiskScorerConfig)
    policy_engine: PolicyEngineConfig = field(default_factory=PolicyEngineConfig)
    attack_classifier: AttackClassifierConfig = field(default_factory=AttackClassifierConfig)

@dataclass(frozen=True)
class TrainingConfig:

    batch_size: int = 256
    num_epochs_ae: int = 50
    num_epochs_lstm: int = 30
    learning_rate_ae: float = 1e-3
    learning_rate_lstm: float = 5e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 7
    device: str = "cpu"
    random_seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False
    gradient_clip_max_norm: float = 1.0

    weight_search_step: float = 0.05

    threshold_search_start: float = 0.05
    threshold_search_end: float = 0.95
    threshold_search_step: float = 0.01
    threshold_optimisation_objective: str = "f1"
    threshold_optimisation_constraint: float = 0.01
    alert_budget: int = 350

    cold_start_min_history_days: Dict[str, float] = field(default_factory=lambda: {
        "corporate_employee": 7.0,
        "factory_operator": 3.0,
        "service_account": 1.0,
    })
    cold_start_w_ae: float = 0.05
    cold_start_w_lstm: float = 0.10
    cold_start_w_policy: float = 0.85

    drift_window_size: int = 1000
    drift_psi_warning: float = 0.1
    drift_psi_alert: float = 0.25
    drift_n_bins: int = 10

@dataclass(frozen=True)
class ProjectConfig:

    paths: PathConfig = field(default_factory=PathConfig)
    data_gen: DataGenConfig = field(default_factory=DataGenConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def get_project_config() -> ProjectConfig:
    config = ProjectConfig()
    config.paths.ensure_dirs()
    logger.info("ProjectConfig initialised. Root: %s", config.paths.project_root)
    return config

PROJECT_CONFIG: ProjectConfig = get_project_config()
