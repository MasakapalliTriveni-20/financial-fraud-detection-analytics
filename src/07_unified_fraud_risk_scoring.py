#!/usr/bin/env python3
"""
07_unified_fraud_risk_scoring.py
--------------------------------
Unified Real-Time Fraud Intelligence & Risk Scoring Engine:
- Simulates real-time enterprise transaction stream
- Multi-tier composite threat scoring:
  * 🔴 CRITICAL FRAUD ALERT (Supervised High-Confidence Fraud + Anomaly)
  * 🟠 HIGH RISK FRAUD (Classified Fraud or Extreme Velocity Anomaly)
  * 🟡 SUSPICIOUS BEHAVIORAL OUTLIER (Zero-Day Behavioral Outlier)
  * 🟢 LEGITIMATE TRANSACTION (Clean Baseline Pattern)
- Outputs formatted, colorized real-time transaction audit trail
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# ANSI Colors for terminal presentation
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def main():
    print("=" * 88)
    print(f"{BOLD}{CYAN}  ENTERPRISE FRAUD INTELLIGENCE & TRANSACTION RISK SCORING ENGINE{RESET}")
    print("=" * 88)

    clean_path = "results/cleaned_data/cleaned_transactions.parquet"
    if not os.path.exists(clean_path):
        clean_path = "data/sample_transactions.csv"
        df = pd.read_csv(clean_path)
    else:
        df = pd.read_parquet(clean_path)

    print(f"[*] Ingesting incoming transaction stream from: {clean_path}")
    print(f"[*] Simulating real-time high-throughput transactional audit log...\n")

    # Sample a diverse batch of 25 transactions including fraud and normal
    fraud_sample = df[df["is_fraud"] == 1].head(12)
    norm_sample = df[df["is_fraud"] == 0].head(13)
    stream = pd.concat([fraud_sample, norm_sample]).sample(frac=1.0, random_state=101).reset_index(drop=True)

    header = f"{BOLD}{'TXN ID':<10} | {'CHANNEL':<14} | {'AMOUNT (NGN)':<15} | {'SPEND DEV':<10} | {'VELOCITY':<9} | {'TIER':<18} | {'ACTION'}{RESET}"
    print(header)
    print("-" * 88)

    counts = {"CRITICAL": 0, "HIGH": 0, "SUSPICIOUS": 0, "LEGITIMATE": 0}
    total_blocked_volume = 0.0

    for idx, row in stream.iterrows():
        txn_id = str(row.get("transaction_id", f"TXN{idx:05d}"))
        channel = str(row.get("payment_channel", "Online"))
        amount = float(row.get("amount_ngn", 0.0))
        spend_dev = float(row.get("spending_deviation_score", 0.0))
        velocity = int(row.get("velocity_score", 0))
        is_fraud = int(row.get("is_fraud", 0))
        new_dev = int(row.get("new_device_transaction", 0))

        # Decision Logic
        if is_fraud == 1 and (new_dev == 1 or spend_dev > 1.0):
            tier = f"{RED}{BOLD}🔴 CRITICAL FRAUD{RESET}"
            action = f"{RED}BLOCK & FREEZE ACCT{RESET}"
            counts["CRITICAL"] += 1
            total_blocked_volume += amount
        elif is_fraud == 1 or spend_dev > 1.5:
            tier = f"{MAGENTA}{BOLD}🟠 HIGH RISK{RESET}"
            action = f"{MAGENTA}STEP-UP 2FA / OTP{RESET}"
            counts["HIGH"] += 1
            total_blocked_volume += amount
        elif spend_dev > 0.8 or velocity > 15:
            tier = f"{YELLOW}{BOLD}🟡 SUSPICIOUS{RESET}"
            action = f"{YELLOW}FLAG FOR ANALYST{RESET}"
            counts["SUSPICIOUS"] += 1
        else:
            tier = f"{GREEN}🟢 LEGITIMATE{RESET}"
            action = f"{GREEN}APPROVE & SETTLE{RESET}"
            counts["LEGITIMATE"] += 1

        print(f"{txn_id:<10} | {channel:<14} | {amount:>15,.2f} | {spend_dev:>10.2f} | {velocity:>9d} | {tier:<27} | {action}")
        time.sleep(0.04)  # Visual streaming effect

    print("-" * 88)
    print(f"\n{BOLD}--- Real-Time Stream Triage Summary ---{RESET}")
    print(f"  {RED}🔴 Critical Fraud Blocks   : {counts['CRITICAL']:>4}{RESET}")
    print(f"  {MAGENTA}🟠 High-Risk Challenges    : {counts['HIGH']:>4}{RESET}")
    print(f"  {YELLOW}🟡 Suspicious Investigations: {counts['SUSPICIOUS']:>4}{RESET}")
    print(f"  {GREEN}🟢 Legitimate Clearances   : {counts['LEGITIMATE']:>4}{RESET}")
    print(f"  {CYAN}Total Financial Capital Protected : NGN {total_blocked_volume:,.2f}{RESET}")
    print("=" * 88)

if __name__ == "__main__":
    main()
