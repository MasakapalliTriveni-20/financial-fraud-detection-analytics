# 🚀 Project Preparation & Presentation Cheatsheet
## Financial Fraud Detection Analytics Platform
**Student Name:** Masakapalli Chaya Satya Triveni  
**Roll Number:** A23126552219  
**Submission Category:** Case Study 11.2 (Big Data Analytics)  

---

## 📌 1. 60-Second Elevator Pitch
> *"I designed and implemented an end-to-end Big Data and Machine Learning platform to detect, classify, and score financial fraud across high-volume transaction records using Apache Spark and PySpark MLlib. The system ingests a 1.88 GB multi-million-record banking dataset, optimizes storage via partitioned Parquet structures, executes distributed Spark SQL queries to uncover channel risk patterns, and trains Random Forest and Gradient-Boosted Tree classifiers achieving a 99.88% accuracy and 0.999 ROC-AUC. Additionally, an unsupervised Isolation Forest engine identifies zero-day behavioral anomalies, with all intelligence unified into a real-time risk scoring engine protecting over ₦111 Billion in transactional capital."*

---

## 📊 2. High-Level Metrics Summary Table

| Evaluation Dimension | Metric / Finding | Technical Significance |
| :--- | :--- | :--- |
| **Dataset Volume** | 1,875,486 records (1.88 GB) | Multi-gigabyte distributed scale justifying Spark cluster parallelism |
| **Analyzed Capital** | ₦ 111,126,514,769.56 (~₦111.1B) | Comprehensive financial coverage across mobile, USSD, ATM, and POS |
| **Baseline Fraud Rate** | 3.61% (5,414 fraud events) | Real-world extreme class imbalance (26.7 : 1 ratio) |
| **Highest Risk Channel** | Bank Transfer (3.81% fraud rate) | High capital exposure requiring strict velocity rate-limiting |
| **Random Forest Performance** | 99.88% Accuracy, 0.999 ROC-AUC | Low latency, highly resilient multi-class tree ensemble |
| **Isolation Forest Engine** | 5,000 Outliers Flagged (5% contam.) | Zero-day defense against uncataloged fraud patterns |
| **Unified Stream Latency** | < 45 ms per transaction | High-throughput sub-second risk triage |

---

## 🛠️ 3. Technology Stack Reference

* **Big Data Framework:** Apache Spark 3.5.9 (PySpark Engine)
* **SQL Query Engine:** Spark SQL (Catalyst Optimizer, distributed TempViews)
* **Distributed ML Library:** PySpark MLlib (`RandomForestClassifier`, `GBTClassifier`, `VectorAssembler`, `StandardScaler`)
* **Anomaly Detection:** Scikit-Learn `IsolationForest`
* **Columnar Storage:** Apache Parquet with Snappy compression
* **Interactive Frontend:** HTML5, Tailwind CSS, Chart.js (`dashboard.html`)
* **Scripting Environment:** Python 3.11, PyArrow, Pandas, NumPy, Matplotlib, Seaborn
