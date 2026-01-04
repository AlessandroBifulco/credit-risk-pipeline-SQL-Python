import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

# 1. DATABASE SETUP & ETL

db_name = 'home_credit.db'
conn = sqlite3.connect(db_name)

# Mapping Kaggle CSV files to cleaner SQL Table names
file_mapping = {
    'application_train.csv': 'applications',   # Main client data
    'bureau.csv': 'bureau_history'             # History of previous loans
}

for csv_file, table_name in file_mapping.items():
    if os.path.exists(csv_file):
        print(f"Reading file: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Data Cleaning: Sanitize column names for SQL (remove spaces, special chars)
        df.columns = [c.replace(' ', '_').replace(':', '').replace('-', '_') for c in df.columns]
        
        print(f"Writing table '{table_name}' into database...")
        
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"Table '{table_name}' created successfully with {len(df)} rows.")
    else:
        print(f"WARNING: File '{csv_file}' not found. Please check directory.")

# 2. FEATURE ENGINEERING

df_sample = pd.read_sql_query("SELECT * FROM applications LIMIT 5", conn)
print("Columns available:", list(df_sample.columns))

conn.close()
conn = sqlite3.connect('home_credit.db')

sql_query = """
SELECT
    APP.SK_ID_CURR,
    APP.TARGET,
    APP.NAME_CONTRACT_TYPE,
    APP.CODE_GENDER,
    APP.AMT_INCOME_TOTAL,
    APP.AMT_CREDIT as CURRENT_LOAN_AMOUNT,
    APP.AMT_ANNUITY,
    
    COALESCE(B.BUREAU_LOAN_COUNT, 0) as PAST_LOANS_COUNT,
    COALESCE(B.TOTAL_PAST_DEBT, 0) as TOTAL_PAST_DEBT,
    COALESCE(B.ACTIVE_LOANS, 0) as ACTIVE_LOANS_COUNT,
    COALESCE(B.AVG_PAST_CREDIT_LIMIT, 0) as AVG_PAST_CREDIT_LIMIT,
    
    APP.AMT_CREDIT / NULLIF(APP.AMT_INCOME_TOTAL, 0) as CREDIT_TO_INCOME_RATIO

FROM applications APP

LEFT JOIN (
    SELECT
        SK_ID_CURR,
        COUNT(*) as BUREAU_LOAN_COUNT,
        SUM(AMT_CREDIT_SUM_DEBT) as TOTAL_PAST_DEBT,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) as ACTIVE_LOANS,
        AVG(AMT_CREDIT_SUM) as AVG_PAST_CREDIT_LIMIT
    FROM bureau_history
    GROUP BY SK_ID_CURR
) B 
ON APP.SK_ID_CURR = B.SK_ID_CURR

WHERE APP.TARGET IS NOT NULL
LIMIT 10000;
"""

df_final = pd.read_sql_query(sql_query, conn)
print(f"Dataset loaded: {df_final.shape[0]} rows.")
print(df_final.head())

conn.close() 

# 3. PREPROCESSING & SPLIT

# Drop rows where Target is missing
df_clean = df_final.dropna(subset=['TARGET']).copy()

features = [
    'AMT_INCOME_TOTAL', 
    'CURRENT_LOAN_AMOUNT', 
    'AMT_ANNUITY', 
    'PAST_LOANS_COUNT', 
    'TOTAL_PAST_DEBT', 
    'ACTIVE_LOANS_COUNT', 
    'CREDIT_TO_INCOME_RATIO'
]

X = df_clean[features]
y = df_clean['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} clients")
print(f"Test set size:     {X_test.shape[0]} clients")

# 4. MACHINE LEARNING PIPELINE

model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler()),                  
    ('classifier', LogisticRegression(class_weight='balanced', C=1.0, solver='liblinear')) 
])

model_pipeline.fit(X_train, y_train)

# 5. EVALUATION & METRICS

print(">>> Evaluating Performance:")
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

roc_score = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC SCORE: {roc_score:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, columns=['Pred_Repaid', 'Pred_Default'], index=['Actual_Repaid', 'Actual_Default']))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. EXPLAINABILITY (Coefficient Analysis)

# Extract weights (betas) to understand feature impact
coeffs = model_pipeline.named_steps['classifier'].coef_[0]
feature_importance = pd.DataFrame({'Feature': features, 'Weight': coeffs})
feature_importance = feature_importance.sort_values(by='Weight', ascending=False)

print("\nWhich factors increase risk?")
print(feature_importance)

# 7. PLOTTING ROC CURVE

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--') # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credit Risk Model')
plt.legend()
plt.show()