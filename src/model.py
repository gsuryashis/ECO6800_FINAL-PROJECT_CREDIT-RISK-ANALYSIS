import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
WOE_DATA_PATH = 'loan_data_woe_transformed.csv'
MODEL_ARTIFACTS_PATH = 'scorecard_artifacts.pkl'
TARGET_VARIABLE = 'default_status'

# --- Main Script ---

# 1. Load WoE Transformed Data
print(">>> STEP 1: LOADING WoE TRANSFORMED DATA...")
df_woe = pd.read_csv(WOE_DATA_PATH)
print(f"Loaded data with shape: {df_woe.shape}")

# 2. Split Data for Training and Testing
print("\n>>> STEP 2: SPLITTING DATA INTO TRAINING AND TESTING SETS...")
X = df_woe.drop(columns=[TARGET_VARIABLE])
y = df_woe[TARGET_VARIABLE]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 3. Train Logistic Regression Model
print("\n>>> STEP 3: TRAINING LOGISTIC REGRESSION MODEL...")
log_reg = LogisticRegression(solver='lbfgs', C=0.1, random_state=42)
log_reg.fit(X_train, y_train)
print("Model training complete.")

# 4. Evaluate Model Performance
print("\n>>> STEP 4: EVALUATING MODEL PERFORMANCE...")
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# AUC Score
auc = metrics.roc_auc_score(y_test, y_pred_proba)
# Gini Coefficient
gini = 2 * auc - 1

print(f"  - AUC Score on Test Set: {auc:.4f}")
print(f"  - Gini Coefficient on Test Set: {gini:.4f}")

# Plot ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# 5. Save Model and Column List
print("\n>>> STEP 5: SAVING MODEL ARTIFACTS...")
# We need to save the trained model AND the list of WoE columns it expects
model_artifacts = {
    'model': log_reg,
    'model_columns': X_train.columns.tolist() # The list of `_woe` columns
}

with open(MODEL_ARTIFACTS_PATH, 'wb') as f:
    pickle.dump(model_artifacts, f)

print(f"✅ Model and required columns saved to '{MODEL_ARTIFACTS_PATH}'")
print("\n--- SCRIPT COMPLETE ---")