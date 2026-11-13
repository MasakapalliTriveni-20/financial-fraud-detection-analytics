#!/usr/bin/env python3
"""
01_dataset_exploration.py
-------------------------
Large-Scale Financial Dataset Exploration & Schema Inspection
Analyzes the 1.88 GB Financial Transactions & Fraud Detection dataset:
- File metadata & memory footprint
- Complete schema & column profiling
- Fraud class distribution & class imbalance metrics
- Missing value audit
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure consistent PySpark / Python environment
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def main():
    print("=" * 80)
    print("  FINANCIAL FRAUD DETECTION ANALYTICS: DATASET EXPLORATION & PROFILING")
    print("=" * 80)

    # Determine input dataset path (check raw full dataset first, fallback to sample)
    raw_path = "data/transactions_raw.csv"
    if not os.path.exists(raw_path):
        raw_path = "data/sample_transactions.csv"

    file_size_bytes = os.path.getsize(raw_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_bytes / (1024 * 1024 * 1024)

    print(f"[*] Dataset Target File : {raw_path}")
    print(f"[*] Raw File Size       : {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")

    # Read a sample chunk for structural profiling
    print("\n[*] Inspecting Schema & Feature Types (first 50,000 records)...")
    sample_df = pd.read_csv(raw_path, nrows=50000)

    total_cols = len(sample_df.columns)
    print(f"[*] Total Columns Detected : {total_cols}")
    print("\n--- Column Catalog & Data Types ---")
    for i, (col, dtype) in enumerate(sample_df.dtypes.items(), 1):
        null_count = sample_df[col].isnull().sum()
        null_pct = (null_count / len(sample_df)) * 100
        print(f"  {i:2d}. {col:<32} : {str(dtype):<10} | Nulls: {null_count:>5} ({null_pct:.1f}%)")

    # Target class analysis
    target_col = "is_fraud"
    if target_col in sample_df.columns:
        print("\n--- Target Variable Distribution (`is_fraud`) ---")
        val_counts = sample_df[target_col].value_counts(dropna=False)
        for label, count in val_counts.items():
            pct = (count / len(sample_df)) * 100
            label_name = "FRAUDULENT (1)" if str(label).lower() in ["true", "1"] else "LEGITIMATE (0)"
            print(f"  {label_name:<20} : {count:>6} transactions ({pct:6.2f}%)")

    # Financial Transaction Metrics
    if "amount_ngn" in sample_df.columns:
        amt = sample_df["amount_ngn"]
        print("\n--- Financial Volume Summary (Currency Units) ---")
        print(f"  Mean Amount     : {amt.mean():>15,.2f}")
        print(f"  Median Amount   : {amt.median():>15,.2f}")
        print(f"  Min Amount      : {amt.min():>15,.2f}")
        print(f"  Max Amount      : {amt.max():>15,.2f}")
        print(f"  Std Deviation   : {amt.std():>15,.2f}")

    # Payment Channels Breakdown
    if "payment_channel" in sample_df.columns:
        print("\n--- Payment Channel Frequency ---")
        channels = sample_df["payment_channel"].value_counts()
        for ch, cnt in channels.items():
            print(f"  {str(ch):<16} : {cnt:>6} ({cnt / len(sample_df) * 100:5.1f}%)")

    print("\n[+] Dataset exploration and initial profiling completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
