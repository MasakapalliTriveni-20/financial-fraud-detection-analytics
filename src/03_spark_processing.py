#!/usr/bin/env python3
"""
03_spark_processing.py
----------------------
Apache Spark Distributed ETL & Columnar Parquet Partitioning
- Initializes optimized local Apache Spark Session
- Loads preprocessed transactions into distributed PySpark DataFrame
- Implements distributed schema casting and registers SQL TempView
- Partitions large datasets across `payment_channel` for low-latency querying
- Outputs distributed Parquet partitions to `results/parquet/`
"""

import os
import sys

# PySpark Environment Configuration
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, sum as spark_sum

def main():
    print("=" * 80)
    print("  APACHE SPARK DISTRIBUTED PROCESSING & PARQUET CONVERSION")
    print("=" * 80)

    # 1. Initialize Spark Session
    print("[*] Initializing Apache Spark Session (Local Distributed Cluster)...")
    spark = SparkSession.builder \
        .appName("FinancialFraud_DistributedETL") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"[+] Apache Spark Version : {spark.version}")
    print(f"[+] Spark Master         : {spark.sparkContext.master}")
    print(f"[+] Application ID       : {spark.sparkContext.applicationId}")

    # 2. Ingest Data
    parquet_input = "results/cleaned_data/cleaned_transactions.parquet"
    if not os.path.exists(parquet_input):
        parquet_input = "data/sample_transactions.csv"
        print(f"[*] Ingesting raw CSV dataset into Spark: {parquet_input}")
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(parquet_input)
    else:
        print(f"[*] Ingesting cleaned columnar dataset into Spark: {parquet_input}")
        df = spark.read.parquet(parquet_input)

    total_records = df.count()
    print(f"[+] Total Transactions Ingested by Spark Engine: {total_records:,}")

    # 3. Print Spark Schema
    print("\n--- Spark Distributed DataFrame Schema ---")
    df.printSchema()

    # 4. Register SQL TempView for distributed querying
    df.createOrReplaceTempView("financial_transactions")
    print("[+] Registered SQL TempView `financial_transactions` for distributed analytics.")

    # 5. Distributed Partitioning & Parquet Export
    output_parquet_dir = "results/parquet/partitioned_transactions"
    print(f"\n[*] Writing partitioned Parquet files by `payment_channel` -> {output_parquet_dir} ...")
    df.write \
        .mode("overwrite") \
        .partitionBy("payment_channel") \
        .parquet(output_parquet_dir)

    print(f"[+] Distributed Parquet partitioning completed successfully!")

    # Verify partition directories
    if os.path.exists(output_parquet_dir):
        print("\n--- Generated Partition Directories ---")
        for item in sorted(os.listdir(output_parquet_dir)):
            if not item.startswith("."):
                print(f"  ├── {item}")

    spark.stop()
    print("\n[+] Apache Spark ETL and Partitioning pipeline finished successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
