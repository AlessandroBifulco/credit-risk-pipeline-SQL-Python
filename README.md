# Credit Risk Pipeline SQL Python

## Project Overview
This project simulates a real-world **Credit Risk Department** workflow. The goal is to predict the probability of a client defaulting on a loan (Target = 1) by analyzing demographic data and historical credit bureau records.

This solution implements a **Hybrid Architecture**:
1.  **SQL (SQLite)** is used as a Data Warehouse for robust **ETL** and complex **Feature Engineering** (Joins, Aggregations, Window Functions);
2.  **Python (Scikit-Learn)** is used for the **Machine Learning Pipeline**, statistical inference, and evaluation.


### Data Download
Due to GitHub's file size limits (>100MB), the dataset is not included in this repository. You must download the source files directly from Kaggle;
1.  * **Source:** [Home Credit Default Risk - Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk/data)
2.  * **Required Files:** `application_train.csv`, `bureau.csv`;
3.  * **Instruction:** Download these files, unzip them, and place them in the root directory of the project (the same folder where the .py script is located).

## Business Objective
* **Goal:** Automate the loan approval process by identifying high-risk applicants.
* **Impact:** Minimize Non-Performing Loans (NPLs) while maximizing the approval rate for solvable clients.
* **Key Metric:** **ROC-AUC** (Area Under the Curve) to measure the model's ranking ability, prioritizing Recall (catching defaulters) over Precision.

---

## Tech Stack & Libraries

| Library | Purpose | Why I used it? |
| :--- | :--- | :--- |
| **`sqlite3`** | Database Management | To simulate a production environment where data lives in a Relational DB, not flat files. Used for complex joins that are memory-intensive in Pandas. |
| **`pandas`** | Data Manipulation | To bridge the gap between the SQL database and the ML engine. |
| **`scikit-learn`** | Machine Learning | Built a robust **`Pipeline`** (Imputer → Scaler → Model) to prevent data leakage and ensure reproducibility. |
| **`matplotlib`** | Visualization | To plot the **ROC Curve** and visualize the trade-off between True Positives and False Positives. |

---

## Methodology

### 1. Data Ingestion (ETL)
* Raw data (`application_train.csv`, `bureau.csv`) is ingested into a local **SQLite** database.
* Column names are sanitized to be SQL-compliant.

### 2. SQL Feature Engineering
Instead of using Python for everything, I leveraged SQL to create an **Analytical Base Table (ABT)**.
* **Aggregations:** Calculated `TOTAL_PAST_DEBT` and `ACTIVE_LOANS` from the Bureau history table using `GROUP BY` and `CASE WHEN`.
* **Derived Metrics:** Calculated `CREDIT_TO_INCOME_RATIO` directly in SQL using `NULLIF` to handle division-by-zero errors.

### 3. Machine Learning Pipeline
I implemented a **Logistic Regression** model because explainability is crucial in Finance (we need to explain *why* a loan was rejected).
* **Imputation:** Median strategy (robust to outliers in financial data).
* **Scaling:** Z-Score Normalization (`StandardScaler`) to compare income (millions) and age (tens).
* **Class Weighting:** Used `class_weight='balanced'` to handle the dataset imbalance (Defaults are rare events).

---

## Results & Key Insights

### Performance Metrics
* **ROC-AUC Score:** `0.63` (Baseline Model).
    * *Interpretation:* The model successfully identifies risk drivers. It performs significantly better than random guessing (0.50).
* **Recall (Target=1):** `0.64`.
    * *Interpretation:* The model captures **64%** of actual defaulters. This "Safety-First" approach minimizes financial loss for the bank.

### Feature Importance (Drivers of Risk)
The model identified the following behaviors as the strongest predictors:
1.  **High Risk:** Clients with many **Active Loans** (`ACTIVE_LOANS_COUNT` weight: `+0.60`);
2.  **High Risk:** Clients with high annual installments (`AMT_ANNUITY`);
3.  **Low Risk:** Clients with a long credit history (`PAST_LOANS_COUNT` weight: `-0.53`).

---

### Author

Alessandro Bifulco
