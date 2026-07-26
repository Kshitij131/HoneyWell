# Behavioral Anomaly Detection Platform

AI-Powered Behavioral Anomaly Detection for IT/OT environments. Detects
insider threats, credential attacks, lateral movement, data exfiltration,
and device spoofing using a hybrid ML pipeline.

## Architecture

![System Architecture](Architecture%20Diagram.png)

### Hybrid Detection Pipeline

| Component | Purpose |
|-----------|---------|
| **SharedAutoencoder** | Reconstruction-based anomaly detection across all personas |
| **Adaptive Threshold** | Per-entity-type 95th percentile thresholds |
| **LSTMSequenceClassifier** | Temporal sequence classification with Focal Loss |
| **PolicyEngine** | Configurable rule-based scoring (10 rules) |
| **RiskScorer** | Budgeted weighted combination: `R = 0.10·AE + 0.80·LSTM + 0.10·Policy`; thresholds are global 0.37, corporate 0.73, factory 0.37, service 0.42. |
| **Signature rules** | Precision-first impossible-travel, device-spoofing, and credential-stuffing signatures, unioned with the ML threshold. |
| **Captum IG** | Integrated Gradients explainability |

## Configuration

Final calibration no longer uses the old hard 0.15 per-attack recall veto during threshold search. The shipped path uses a saved-model calibration stage that tunes `w_ae`, `w_lstm`, and `w_policy` with a budget-constrained macro-F1 objective on the validation split, then selects global and persona thresholds under the same alert-budget framing. Rare, high-precision classes are handled through the signature-rule fast path instead of forcing one global ML threshold to satisfy incompatible objectives.

### Attack Types Detected

| Attack | MITRE Technique | Default Weight |
|--------|----------------|----------------|
| Brute Force | T1110 | 35% |
| Credential Stuffing | T1110.004 | 25% |
| Impossible Travel | T1078 | 15% |
| Device Spoofing | T1036 | 10% |
| Lateral Movement | T1021 | 8% |
| Low-and-Slow Exfiltration | T1048 | 5% |
| Insider Drift | T1074 | 2% |

### Persona Types

- **Corporate Employee** — Office/VPN/Home ISP subnets, MFA/SSO/Password auth
- **Factory Operator** — Plant/OT subnets, Badge/Password/Certificate auth
- **Service Account/IoT** — Static infrastructure IPs, Certificate/API Key/Token auth

### Known Limitations

- Insider drift is detected reliably, but its attack-type label may be confused with lateral movement: on true insider-drift alerts the prior softmax audit averaged 0.77 for lateral movement versus 0.17 for insider drift. Detection and classification should therefore be interpreted separately.
- The impossible-travel signature intentionally favors sudden, externally sourced implausible jumps. It may miss slow, sustained location drift; this is a deliberate precision/recall trade-off.
- Credential stuffing remains the disclosed detection gap: the synthetic attack generator rotates external source IPs, so a precision-first “few source IPs across many accounts” signature has zero validation recall under its single-digit false-positive constraint. The ML path detects 41.57% of test credential-stuffing events.
- `cold_start_demo_user` and `chain_demo_user` are clearly labelled test-only synthetic demo entities. `chain_demo_user` exists solely to exercise genuine temporal chain reconstruction from correlated alert events; no chain ID is injected into the output.

### Alert Confidence Semantics

`classification_confidence` is the attack-type classifier’s softmax probability only when the classifier selected the final attack label for an `ml_threshold` alert. Signature-selected labels take precedence when a signature fires (including `both`); the dashboard deliberately shows “Rule-based (validated)” or “Signature override (validated)” instead of attaching an unrelated classifier probability. This is classification confidence, not a separate detection-confidence score.

### Concept Drift & Retraining Policy

The pipeline implements a two-tier Population Stability Index (PSI) drift monitor evaluating the distribution of Autoencoder reconstruction errors for each persona.

- **Warning Tier (PSI ≥ 0.10)**: Moderate distribution shift detected. An alert is shown on the dashboard. No automated retraining is triggered, but SOC teams should investigate for gradual behavioral changes.
- **Alert Tier (PSI ≥ 0.25)**: Significant distribution shift detected. Retraining should be triggered to update the baseline representations of normality.

## Project Structure

```
behavioral_anomaly_detection/
├── data/
│   ├── raw/              # raw_logs.csv, ground_truth.csv
│   └── processed/        # Feature arrays, scaler, encoder, metadata
├── models/               # Trained model checkpoints, thresholds
├── outputs/              # Alerts, evaluation reports
├── config.py             # All configuration (paths, seeds, hyperparameters)
├── data_generator.py     # Synthetic data generation with attack injection
├── feature_pipeline.py   # Point-in-time feature engineering
├── models.py             # PyTorch model definitions and training
├── evaluation.py         # Metrics computation and report generation
├── alert_engine.py       # Alert generation, MITRE mapping, chain reconstruction
├── inference_pipeline.py # End-to-end inference orchestration
├── pipeline_runner.py    # CLI entry point for all stages
├── dashboard.py          # Streamlit SOC dashboard
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd behavioral_anomaly_detection
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python pipeline_runner.py --stage all
```

This runs all 4 stages:
1. **generate** — Creates 100k synthetic events with ~1-3% attacks
2. **features** — Engineers 16 point-in-time features, splits, scales
3. **train** — Trains Autoencoder (50 epochs) + LSTM (30 epochs)
4. **infer** — Runs inference, generates alerts, evaluates, reports

### 3. Run Individual Stages

```bash
python pipeline_runner.py --stage generate
python pipeline_runner.py --stage features
python pipeline_runner.py --stage train
python pipeline_runner.py --stage infer --risk-threshold 0.6
```

### 4. Launch Dashboard

```bash
streamlit run dashboard.py
```

## Configuration

All configurable values are in `config.py` as frozen dataclasses:

- **PathConfig** — File/directory paths
- **DataGenConfig** — Data generation parameters, persona configs, attack weights
- **FeatureConfig** — Feature definitions, split ratios, scaler type
- **AutoencoderConfig** — AE architecture (dims, dropout, activation)
- **LSTMConfig** — LSTM architecture (hidden dims, layers, sequence length)
- **FocalLossConfig** — Class imbalance loss (alpha, gamma)
- **RiskScorerConfig** — Hybrid weights (w_ae, w_lstm, w_policy)
- **PolicyEngineConfig** — Rule definitions and thresholds
- **TrainingConfig** — Batch size, epochs, learning rates, early stopping

## Evaluation Metrics

The platform computes and reports:
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix, False Positive Rate
- Per-attack-type performance
- Per-persona performance
- Reconstruction error and risk score distributions
- Inference latency statistics

## Coding Standards

- Python 3.11+
- Explicit type hints everywhere
- Google-style docstrings
- PEP8 compliant
- `logging` module only (no `print()`)
- Deterministic random seeds
- Vectorized NumPy/Pandas operations
- No TODOs, placeholders, or pseudocode

## License

Proprietary. Internal use only.
