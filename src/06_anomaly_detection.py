#!/usr/bin/env python3
"""
06_anomaly_detection.py
-----------------------
Unsupervised Zero-Day Anomaly Detection with Isolation Forest
- Detects novel, unprecedented transaction anomalies without labeled data
- Analyzes multi-dimensional transaction velocity and spending deviations
- Computes continuous anomaly scores & decision function thresholds
- Exports anomaly detection metrics and flagged outlier transactions to `results/anomalies/`
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 80)
    print("  UNSUPERVISED TRANSACTION ANOMALY DETECTION: ISOLATION FOREST")
    print("=" * 80)

    start_time = time.time()
    parquet_path = "results/cleaned_data/cleaned_transactions.parquet"
    if not os.path.exists(parquet_path):
        parquet_path = "data/sample_transactions.csv"
        df = pd.read_csv(parquet_path)
    else:
        df = pd.read_parquet(parquet_path)

    sample_size = min(100000, len(df))
    print(f"[*] Ingesting {sample_size:,} transaction records for unsupervised anomaly profiling...")
    sample_df = df.sample(n=sample_size, random_state=42).copy()

    # Feature extraction for behavioral deviations
    feature_cols = [
        "amount_ngn", "spending_deviation_score", "velocity_score",
        "geo_anomaly_score", "device_seen_count", "ip_seen_count",
        "user_txn_count_total", "user_avg_txn_amt", "txn_count_last_1h",
        "total_amount_last_1h", "channel_risk_score", "persona_fraud_risk"
    ]
    avail_cols = [c for c in feature_cols if c in sample_df.columns]
    print(f"[*] Anomaly Detection Feature Dimensions: {len(avail_cols)}")

    X = sample_df[avail_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configure Isolation Forest
    contamination_rate = 0.05  # Expect ~5% abnormal behavioral deviations
    print(f"[*] Training Isolation Forest (n_estimators=100, contamination={contamination_rate * 100:.1f}%)...")
    iso_start = time.time()
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    iso_train_time = time.time() - iso_start
    print(f"[+] Isolation Forest Training Complete ({iso_train_time:.2f}s)")

    # Predictions (-1 = anomaly, 1 = normal)
    preds = iso_forest.predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)

    sample_df["anomaly_flag"] = np.where(preds == -1, 1, 0)
    sample_df["anomaly_score"] = scores

    anomalies_detected = int(sample_df["anomaly_flag"].sum())
    normal_detected = int(sample_size - anomalies_detected)
    pct_anomalous = (anomalies_detected / sample_size) * 100

    print(f"\n--- Anomaly Detection Results ---")
    print(f"  Total Processed Transactions : {sample_size:>8,}")
    print(f"  Flagged Anomalous Outliers   : {anomalies_detected:>8,} ({pct_anomalous:.2f}%)")
    print(f"  Conforming Baseline Events   : {normal_detected:>8,} ({100 - pct_anomalous:.2f}%)")

    # Correlation with known fraud flags
    if "is_fraud" in sample_df.columns:
        fraud_in_anomalies = sample_df[(sample_df["anomaly_flag"] == 1) & (sample_df["is_fraud"] == 1)]
        detection_precision = len(fraud_in_anomalies) / max(1, anomalies_detected)
        print(f"  Known Fraud in Anomalies     : {len(fraud_in_anomalies):>8,}")
        print(f"  Unsupervised Overlap Rate    : {detection_precision * 100:>7.2f}%")

    output_dir = "results/anomalies"
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics JSON
    metrics = {
        "engine": "Isolation Forest (Scikit-Learn)",
        "sample_size": sample_size,
        "n_estimators": 100,
        "contamination": contamination_rate,
        "anomalies_detected": anomalies_detected,
        "normal_detected": normal_detected,
        "pct_anomalous": round(pct_anomalous, 2),
        "training_time_sec": round(iso_train_time, 2)
    }
    with open(os.path.join(output_dir, "anomaly_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Export top anomalous transactions
    flagged = sample_df[sample_df["anomaly_flag"] == 1].sort_values(by="anomaly_score")
    out_csv = os.path.join(output_dir, "flagged_anomalous_transactions.csv")
    export_cols = [c for c in ["transaction_id", "amount_ngn", "payment_channel", "location", "is_fraud", "anomaly_score", "spending_deviation_score", "velocity_score"] if c in flagged.columns]
    flagged[export_cols].head(5000).to_csv(out_csv, index=False)
    print(f"[+] Flagged anomalies exported -> {out_csv}")

    elapsed = time.time() - start_time
    print(f"[+] Anomaly detection pipeline finished in {elapsed:.2f} seconds.")
    print("=" * 80)

if __name__ == "__main__":
    main()
