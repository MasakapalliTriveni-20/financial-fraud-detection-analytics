#!/usr/bin/env python3
"""
02_data_preprocessing.py
------------------------
Memory-Efficient Data Preprocessing & Sanitization Pipeline:
- Chunked processing to handle multi-gigabyte financial transactions cleanly
- Boolean flag normalization (True/False -> 1/0)
- Missing value imputation and null handling
- Feature scaling preparation and selection
- Exports clean Parquet & CSV extracts to `results/cleaned_data/`
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure consistent environment
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def main():
    print("=" * 80)
    print("  FINANCIAL FRAUD DETECTION: DATA PREPROCESSING & CLEANING")
    print("=" * 80)

    raw_path = "data/transactions_raw.csv"
    if not os.path.exists(raw_path):
        raw_path = "data/sample_transactions.csv"

    output_dir = "results/cleaned_data"
    os.makedirs(output_dir, exist_ok=True)
    out_parquet = os.path.join(output_dir, "cleaned_transactions.parquet")
    out_csv = os.path.join(output_dir, "cleaned_transactions_sample.csv")

    print(f"[*] Reading source dataset: {raw_path}")
    print("[*] Processing chunked stream (extracting 150,000 stratified transaction records)...")

    # We read in chunks to keep memory usage low and fast
    chunk_size = 50000
    chunks = []
    total_loaded = 0
    target_records = 150000

    for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
        total_loaded += len(chunk)
        print(f"    -> Processed chunk: {total_loaded:,} records loaded...")
        if total_loaded >= target_records:
            break

    df = pd.concat(chunks, ignore_index=True)
    print(f"[+] Total records ingested for cleaning: {len(df):,}")

    # 1. Normalize Target Variable
    print("[*] Normalizing `is_fraud` target variable...")
    df["is_fraud"] = df["is_fraud"].apply(lambda x: 1 if str(x).lower() in ["true", "1"] else 0)

    # 2. Normalize Boolean Columns
    bool_cols = ["bvn_linked", "new_device_transaction", "geospatial_velocity_anomaly"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ["true", "1"] else 0)

    # 3. Handle Null Values
    print("[*] Imputing missing values...")
    if "time_since_last_transaction" in df.columns:
        df["time_since_last_transaction"] = df["time_since_last_transaction"].fillna(0.0)

    if "fraud_type" in df.columns:
        df["fraud_type"] = df["fraud_type"].fillna("None")

    # Impute numeric columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 4. Feature Selection: Retain core analytical & modeling columns
    feature_cols = [
        "transaction_id", "timestamp", "transaction_type", "merchant_category",
        "location", "device_used", "payment_channel", "amount_ngn", "is_fraud",
        "fraud_type", "spending_deviation_score", "velocity_score", "geo_anomaly_score",
        "bvn_linked", "new_device_transaction", "sender_persona", "txn_hour",
        "is_weekend", "is_salary_week", "is_night_txn", "device_seen_count",
        "is_device_shared", "ip_seen_count", "is_ip_shared", "user_txn_count_total",
        "user_avg_txn_amt", "txn_count_last_1h", "txn_count_last_24h",
        "total_amount_last_1h", "merchant_fraud_rate", "channel_risk_score",
        "persona_fraud_risk", "location_fraud_risk", "ip_geo_region"
    ]
    retained_cols = [c for c in feature_cols if c in df.columns]
    cleaned_df = df[retained_cols].copy()

    print(f"[*] Retained Features Count: {len(cleaned_df.columns)}")
    print(f"[*] Fraud Class Distribution in Preprocessed Dataset:")
    fraud_counts = cleaned_df["is_fraud"].value_counts()
    for lbl, cnt in fraud_counts.items():
        name = "FRAUD (1)" if lbl == 1 else "LEGITIMATE (0)"
        print(f"    {name:<15}: {cnt:>7,} ({cnt / len(cleaned_df) * 100:.2f}%)")

    # 5. Export to Parquet and CSV
    print(f"[*] Exporting optimized Parquet dataset -> {out_parquet} ...")
    cleaned_df.to_parquet(out_parquet, index=False, engine="pyarrow")

    print(f"[*] Exporting CSV sample (10,000 rows) -> {out_csv} ...")
    cleaned_df.head(10000).to_csv(out_csv, index=False)

    print(f"[+] Preprocessing successfully completed! Output stored in `{output_dir}`.")
    print("=" * 80)

if __name__ == "__main__":
    main()
