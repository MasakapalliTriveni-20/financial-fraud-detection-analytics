#!/usr/bin/env python3
"""
05_fraud_classification.py
--------------------------
Distributed Supervised Machine Learning Pipeline using PySpark MLlib:
- Feature Engineering via Pipeline (StringIndexer, VectorAssembler, StandardScaler)
- Supervised Classification Models:
  1. PySpark MLlib Random Forest Classifier
  2. PySpark MLlib Gradient-Boosted Tree (GBT) Classifier
- Rigorous Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- Confusion Matrix Extraction & Metrics Export to `results/ml/`
"""

import os
import sys
import json
import time

# PySpark Environment Configuration
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

def main():
    print("=" * 80)
    print("  PYSPARK MLLIB: DISTRIBUTED FRAUD CLASSIFICATION PIPELINE")
    print("=" * 80)

    start_time = time.time()

    spark = SparkSession.builder \
        .appName("FinancialFraud_MLlib_Classification") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    input_path = "results/cleaned_data/cleaned_transactions.parquet"
    if not os.path.exists(input_path):
        input_path = "data/sample_transactions.csv"
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    else:
        df = spark.read.parquet(input_path)

    total_count = df.count()
    print(f"[*] Ingested Dataset Size for ML Training: {total_count:,} records")

    # Cast label to double for MLlib
    df = df.withColumn("label", col("is_fraud").cast("double"))

    # Define Feature Sets
    cat_cols = ["transaction_type", "payment_channel", "device_used", "sender_persona"]
    num_cols = [
        "amount_ngn", "spending_deviation_score", "velocity_score", "geo_anomaly_score",
        "bvn_linked", "new_device_transaction", "txn_hour", "is_weekend",
        "is_salary_week", "is_night_txn", "device_seen_count", "is_device_shared",
        "ip_seen_count", "is_ip_shared", "user_txn_count_total", "user_avg_txn_amt",
        "txn_count_last_1h", "txn_count_last_24h", "total_amount_last_1h",
        "merchant_fraud_rate", "channel_risk_score", "persona_fraud_risk",
        "location_fraud_risk"
    ]

    # Available columns check
    existing_cols = df.columns
    cat_cols = [c for c in cat_cols if c in existing_cols]
    num_cols = [c for c in num_cols if c in existing_cols]

    # 1. Feature Engineering Pipeline Stages
    indexers = [
        StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx", handleInvalid="keep")
        for col_name in cat_cols
    ]
    indexed_cat_cols = [f"{c}_idx" for c in cat_cols]

    assembler = VectorAssembler(
        inputCols=num_cols + indexed_cat_cols,
        outputCol="raw_features",
        handleInvalid="keep"
    )

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

    preproc_pipeline = Pipeline(stages=indexers + [assembler, scaler])
    preproc_model = preproc_pipeline.fit(df)
    processed_df = preproc_model.transform(df).select("label", "features")

    # 2. Train / Test Split (80% Training, 20% Testing)
    train_df, test_df = processed_df.randomSplit([0.80, 0.20], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"[*] Training Set Records: {train_count:,} (80%)")
    print(f"[*] Testing Set Records : {test_count:,} (20%)")

    # 3. Model 1: Random Forest Classifier
    print("\n" + "-" * 60)
    print("  TRAINING MODEL 1: RANDOM FOREST CLASSIFIER (PySpark MLlib)")
    print("-" * 60)
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=30,
        maxDepth=10,
        seed=42
    )
    rf_start = time.time()
    rf_model = rf.fit(train_df)
    rf_train_time = time.time() - rf_start
    print(f"[+] Random Forest Training Time: {rf_train_time:.2f} seconds")

    rf_preds = rf_model.transform(test_df)

    # Evaluators
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

    rf_acc = acc_eval.evaluate(rf_preds)
    rf_f1 = f1_eval.evaluate(rf_preds)
    rf_prec = prec_eval.evaluate(rf_preds)
    rf_rec = rec_eval.evaluate(rf_preds)
    rf_auc = auc_eval.evaluate(rf_preds)
    rf_pr = pr_eval.evaluate(rf_preds)

    print(f"  -> Accuracy : {rf_acc * 100:.2f}%")
    print(f"  -> Precision: {rf_prec * 100:.2f}%")
    print(f"  -> Recall   : {rf_rec * 100:.2f}%")
    print(f"  -> F1-Score : {rf_f1 * 100:.2f}%")
    print(f"  -> ROC-AUC  : {rf_auc:.4f}")
    print(f"  -> PR-AUC   : {rf_pr:.4f}")

    # Confusion Matrix for Random Forest
    rf_tp = rf_preds.filter("label = 1.0 AND prediction = 1.0").count()
    rf_tn = rf_preds.filter("label = 0.0 AND prediction = 0.0").count()
    rf_fp = rf_preds.filter("label = 0.0 AND prediction = 1.0").count()
    rf_fn = rf_preds.filter("label = 1.0 AND prediction = 0.0").count()

    print("\n--- Confusion Matrix (Random Forest) ---")
    print(f"  True Negatives (TN)  : {rf_tn:>6,}")
    print(f"  False Positives (FP) : {rf_fp:>6,}")
    print(f"  False Negatives (FN) : {rf_fn:>6,}")
    print(f"  True Positives (TP)  : {rf_tp:>6,}")

    # 4. Model 2: Gradient-Boosted Tree Classifier
    print("\n" + "-" * 60)
    print("  TRAINING MODEL 2: GRADIENT-BOOSTED TREES (GBTClassifier)")
    print("-" * 60)
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=15,
        maxDepth=6,
        seed=42
    )
    gbt_start = time.time()
    gbt_model = gbt.fit(train_df)
    gbt_train_time = time.time() - gbt_start
    print(f"[+] GBT Training Time: {gbt_train_time:.2f} seconds")

    gbt_preds = gbt_model.transform(test_df)

    gbt_acc = acc_eval.evaluate(gbt_preds)
    gbt_f1 = f1_eval.evaluate(gbt_preds)
    gbt_prec = prec_eval.evaluate(gbt_preds)
    gbt_rec = rec_eval.evaluate(gbt_preds)
    gbt_auc = auc_eval.evaluate(gbt_preds)

    print(f"  -> Accuracy : {gbt_acc * 100:.2f}%")
    print(f"  -> Precision: {gbt_prec * 100:.2f}%")
    print(f"  -> Recall   : {gbt_rec * 100:.2f}%")
    print(f"  -> F1-Score : {gbt_f1 * 100:.2f}%")
    print(f"  -> ROC-AUC  : {gbt_auc:.4f}")

    # 5. Export Metrics Summary to JSON
    output_dir = "results/ml"
    os.makedirs(output_dir, exist_ok=True)
    metrics_data = {
        "dataset_name": "Nigerian Financial Transactions and Fraud Detection Dataset",
        "dataset_source": "Hugging Face / ElectricSheepAfrica Consortium",
        "dataset_size_gb": 1.88,
        "total_records_processed": total_count,
        "training_records": train_count,
        "testing_records": test_count,
        "feature_dimensions": len(num_cols) + len(cat_cols),
        "random_forest": {
            "model_type": "RandomForestClassifier (PySpark MLlib)",
            "num_trees": 30,
            "max_depth": 10,
            "training_time_sec": round(rf_train_time, 2),
            "accuracy": round(rf_acc * 100, 2),
            "precision": round(rf_prec * 100, 2),
            "recall": round(rf_rec * 100, 2),
            "f1_score": round(rf_f1 * 100, 2),
            "roc_auc": round(rf_auc, 4),
            "pr_auc": round(rf_pr, 4),
            "confusion_matrix": {
                "true_negative": rf_tn,
                "false_positive": rf_fp,
                "false_negative": rf_fn,
                "true_positive": rf_tp
            }
        },
        "gradient_boosted_trees": {
            "model_type": "GBTClassifier (PySpark MLlib)",
            "max_iter": 15,
            "max_depth": 6,
            "training_time_sec": round(gbt_train_time, 2),
            "accuracy": round(gbt_acc * 100, 2),
            "precision": round(gbt_prec * 100, 2),
            "recall": round(gbt_rec * 100, 2),
            "f1_score": round(gbt_f1 * 100, 2),
            "roc_auc": round(gbt_auc, 4)
        }
    }

    metrics_file = os.path.join(output_dir, "classification_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\n[+] Saved complete evaluation metrics -> {metrics_file}")

    spark.stop()
    elapsed = time.time() - start_time
    print(f"[+] Total MLlib pipeline executed in {elapsed:.2f} seconds.")
    print("=" * 80)

if __name__ == "__main__":
    main()
