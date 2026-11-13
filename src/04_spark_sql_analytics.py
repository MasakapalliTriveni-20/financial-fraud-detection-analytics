#!/usr/bin/env python3
"""
04_spark_sql_analytics.py
-------------------------
Large-Scale Distributed Analytical Querying with Spark SQL
Executes distributed queries to uncover critical financial fraud patterns:
1. Overall Fraud Volume & Total Financial Exposure
2. Channel Risk & Fraud Incidence Rate (USSD, Mobile, Card, Bank Transfer)
3. Hourly Fraud Velocity & Surge Detection
4. High-Risk Merchant Category Profiling
5. Geolocation Anomaly & Device-Sharing Risk Matrix
Exports all aggregated analytical tables to `results/analytics/`
"""

import os
import sys

# PySpark Environment Configuration
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

def main():
    print("=" * 80)
    print("  SPARK SQL DISTRIBUTED ANALYTICS & FRAUD PATTERN DETECTION")
    print("=" * 80)

    spark = SparkSession.builder \
        .appName("FinancialFraud_SparkSQL_Analytics") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    input_path = "results/cleaned_data/cleaned_transactions.parquet"
    if not os.path.exists(input_path):
        input_path = "data/sample_transactions.csv"
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    else:
        df = spark.read.parquet(input_path)

    df.createOrReplaceTempView("financial_transactions")
    output_dir = "results/analytics"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Query 1: Overall Fraud Volume & Financial Loss Exposure
    # -------------------------------------------------------------------------
    print("\n[Query 1] Overall Fraud Distribution & Financial Exposure:")
    q1 = """
    SELECT 
        CASE WHEN is_fraud = 1 THEN 'FRAUDULENT' ELSE 'LEGITIMATE' END AS status,
        COUNT(*) AS total_transactions,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage_of_total,
        ROUND(SUM(amount_ngn), 2) AS total_financial_volume,
        ROUND(AVG(amount_ngn), 2) AS avg_transaction_amount
    FROM financial_transactions
    GROUP BY is_fraud
    """
    df1 = spark.sql(q1)
    df1.show(truncate=False)
    df1.toPandas().to_csv(os.path.join(output_dir, "01_overall_fraud_distribution.csv"), index=False)

    # -------------------------------------------------------------------------
    # Query 2: Payment Channel Risk Distribution
    # -------------------------------------------------------------------------
    print("\n[Query 2] Payment Channel Fraud Incidence & Lost Volume:")
    q2 = """
    SELECT 
        payment_channel,
        COUNT(*) AS total_txns,
        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_txns,
        ROUND(AVG(is_fraud) * 100, 2) AS fraud_incidence_rate_pct,
        ROUND(AVG(channel_risk_score), 3) AS avg_channel_risk,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount_ngn ELSE 0 END), 2) AS total_fraud_loss
    FROM financial_transactions
    GROUP BY payment_channel
    ORDER BY fraud_incidence_rate_pct DESC
    """
    df2 = spark.sql(q2)
    df2.show(truncate=False)
    df2.toPandas().to_csv(os.path.join(output_dir, "02_channel_fraud_risk.csv"), index=False)

    # -------------------------------------------------------------------------
    # Query 3: Hourly Velocity & Surge Detection
    # -------------------------------------------------------------------------
    print("\n[Query 3] Hourly Velocity & Surge Distribution (24-Hour Cycle):")
    q3 = """
    SELECT 
        txn_hour,
        COUNT(*) AS total_txns,
        SUM(is_fraud) AS fraud_count,
        ROUND(AVG(is_fraud) * 100, 2) AS hourly_fraud_rate_pct,
        ROUND(AVG(velocity_score), 2) AS avg_velocity_score,
        ROUND(AVG(spending_deviation_score), 3) AS avg_spending_deviation
    FROM financial_transactions
    GROUP BY txn_hour
    ORDER BY txn_hour ASC
    """
    df3 = spark.sql(q3)
    df3.show(24, truncate=False)
    df3.toPandas().to_csv(os.path.join(output_dir, "03_hourly_fraud_velocity.csv"), index=False)

    # -------------------------------------------------------------------------
    # Query 4: Top High-Risk Merchant Categories
    # -------------------------------------------------------------------------
    print("\n[Query 4] Top 10 High-Risk Merchant Categories:")
    q4 = """
    SELECT 
        merchant_category,
        COUNT(*) AS total_txns,
        SUM(is_fraud) AS fraud_txns,
        ROUND(AVG(is_fraud) * 100, 2) AS fraud_pct,
        ROUND(SUM(amount_ngn), 2) AS total_merchant_volume
    FROM financial_transactions
    GROUP BY merchant_category
    ORDER BY fraud_txns DESC
    LIMIT 10
    """
    df4 = spark.sql(q4)
    df4.show(truncate=False)
    df4.toPandas().to_csv(os.path.join(output_dir, "04_top_merchant_categories.csv"), index=False)

    # -------------------------------------------------------------------------
    # Query 5: Geolocation Anomaly & Device-Sharing Matrix
    # -------------------------------------------------------------------------
    print("\n[Query 5] Device Sharing & Geolocation Anomaly Matrix:")
    q5 = """
    SELECT 
        is_device_shared,
        is_ip_shared,
        new_device_transaction,
        COUNT(*) AS txn_count,
        SUM(is_fraud) AS fraud_count,
        ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct,
        ROUND(AVG(geo_anomaly_score), 3) AS avg_geo_anomaly_score
    FROM financial_transactions
    GROUP BY is_device_shared, is_ip_shared, new_device_transaction
    ORDER BY fraud_rate_pct DESC
    """
    df5 = spark.sql(q5)
    df5.show(truncate=False)
    df5.toPandas().to_csv(os.path.join(output_dir, "05_device_and_geo_risk.csv"), index=False)

    spark.stop()
    print(f"\n[+] Spark SQL analytical insights generated and saved in `{output_dir}/`.")
    print("=" * 80)

if __name__ == "__main__":
    main()
