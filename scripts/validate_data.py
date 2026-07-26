"""CLI dataset validation script checking schema integrity, timestamps, and ground truth alignment.
"""

import pandas as pd
import numpy as np

LOG_PATH = "data/raw/raw_logs.csv"
LABEL_PATH = "data/raw/ground_truth.csv"

logs = pd.read_csv(LOG_PATH)
labels = pd.read_csv(LABEL_PATH)

print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(f"Logs shape      : {logs.shape}")
print(f"Labels shape    : {labels.shape}")

print("\nColumns:")
print(logs.columns.tolist())

print("\nGround Truth Columns:")
print(labels.columns.tolist())

print("\n" + "=" * 80)
print("1. MISSING VALUES")
print("=" * 80)

print("\nLogs:")
print(logs.isna().sum())

print("\nLabels:")
print(labels.isna().sum())

print("\n" + "=" * 80)
print("2. TIMESTAMP CHECK")
print("=" * 80)

logs["timestamp"] = pd.to_datetime(logs["timestamp"])
labels["timestamp"] = pd.to_datetime(labels["timestamp"])

print("Logs sorted chronologically:",
      logs["timestamp"].is_monotonic_increasing)

print("Labels sorted chronologically:",
      labels["timestamp"].is_monotonic_increasing)

print("\n" + "=" * 80)
print("3. LABEL ALIGNMENT")
print("=" * 80)

print("Entity IDs aligned:",
      (logs["entity_id"] == labels["entity_id"]).all())

print("Timestamps aligned:",
      (logs["timestamp"] == labels["timestamp"]).all())

print("\n" + "=" * 80)
print("4. PERSONA DISTRIBUTION")
print("=" * 80)

print(logs["entity_type"].value_counts())

print("\n" + "=" * 80)
print("5. SESSION DURATION BY PERSONA")
print("=" * 80)

print(
    logs.groupby("entity_type")["session_duration"].describe()
)

print("\n" + "=" * 80)
print("6. AUTH METHODS")
print("=" * 80)

print(
    logs.groupby("entity_type")["auth_method"].value_counts()
)

print("\n" + "=" * 80)
print("7. UNIQUE RESOURCES PER PERSONA")
print("=" * 80)

print(
    logs.groupby("entity_type")["resource_accessed"].nunique()
)

print("\n" + "=" * 80)
print("8. TOP RESOURCES")
print("=" * 80)

print(
    logs.groupby("entity_type")["resource_accessed"]
    .value_counts()
    .head(20)
)

print("\n" + "=" * 80)
print("9. DEVICE FINGERPRINTS")
print("=" * 80)

print(
    logs.groupby("entity_type")["device_fingerprint"]
    .value_counts()
)

print("\n" + "=" * 80)
print("10. ATTACK DISTRIBUTION")
print("=" * 80)

print(labels["attack_type"].value_counts())

print("\nPercentage:")
print(
    labels["attack_type"]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

print("\n" + "=" * 80)
print("11. SAMPLE ATTACK EVENTS")
print("=" * 80)

for attack in labels["attack_type"].unique():
    if attack == "none":
        continue

    print(f"\n----- {attack} -----")

    idx = labels[labels.attack_type == attack].index[:3]

    print(logs.loc[idx][[
        "entity_id",
        "entity_type",
        "timestamp",
        "source_ip",
        "geo_location",
        "resource_accessed",
        "auth_method",
        "device_fingerprint"
    ]])

print("\n" + "=" * 80)
print("12. UNIQUE ENTITIES")
print("=" * 80)

print("Unique entities:", logs["entity_id"].nunique())

print("\nEntities by type:")

print(
    logs.groupby("entity_type")["entity_id"]
    .nunique()
)

print("\n" + "=" * 80)
print("13. COMMAND SEQUENCE EXAMPLES")
print("=" * 80)

for persona in logs["entity_type"].unique():

    print(f"\n{persona}")

    print(
        logs.loc[
            logs.entity_type == persona,
            "command_sequence"
        ].head(5).tolist()
    )

print("\n" + "=" * 80)
print("14. DUPLICATE ROW CHECK")
print("=" * 80)

print("Duplicate log rows:",
      logs.duplicated().sum())

print("\n" + "=" * 80)
print("15. BASIC VALIDATION")
print("=" * 80)

print("Row counts equal:",
      len(logs) == len(labels))

print("Attack rate:",
      round(labels["is_attack"].mean() * 100, 3), "%")

print("\nValidation Complete.")
