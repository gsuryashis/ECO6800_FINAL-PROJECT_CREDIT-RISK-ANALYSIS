"""
main.py — ECO 6800 Credit Risk Analysis Pipeline
Run: uv run main.py

Writes three files to outputs/:
  baseline_metric.json    — vanilla logistic regression ROC-AUC
  primary_metric.json     — LightGBM ROC-AUC (primary success criterion)
  milestone_manifest.json — summary of all outputs produced
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "raw", "lc_loan.csv")
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & build target
# ---------------------------------------------------------------------------
print("Loading data …")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ Data file not found at {DATA_PATH!r}.\n"
        "Download lc_loan.csv from Google Drive / Kaggle and place it there.\n"
        "See README.md for full instructions."
    )

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Keep only terminal loan statuses (exclude right-censored)
keep_statuses = {"Charged Off", "Default", "Late (31-120 days)", "Fully Paid"}
df = df[df["loan_status"].isin(keep_statuses)].copy()

# Binary target: 1 = bad (default), 0 = good (fully paid)
df["defaultstatus"] = (
    df["loan_status"].isin({"Charged Off", "Default", "Late (31-120 days)"})
).astype(int)

print(f"  After filtering: {df.shape[0]:,} rows")
print(f"  Bad rate: {df['defaultstatus'].mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# 2. Feature selection
#    Use numeric application-time features only; drop post-approval leakage.
# ---------------------------------------------------------------------------
LEAKAGE_COLS = {
    "loan_status", "defaultstatus",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
    "last_pymnt_d", "next_pymnt_d", "total_pymnt", "total_pymnt_inv",
    "out_prncp", "out_prncp_inv",
}

numeric_features = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in LEAKAGE_COLS
]

X = df[numeric_features].copy()
y = df["defaultstatus"].copy()

# Impute with column median
X = X.fillna(X.median(numeric_only=True))

print(f"  Features used: {len(numeric_features)}")

# ---------------------------------------------------------------------------
# 3. 80/20 stratified split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# 4. Baseline — vanilla logistic regression
# ---------------------------------------------------------------------------
print("\nTraining baseline (Logistic Regression) …")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

baseline_lr = LogisticRegression(max_iter=500, random_state=42)
baseline_lr.fit(X_train_sc, y_train)

baseline_proba = baseline_lr.predict_proba(X_test_sc)[:, 1]
baseline_auc = roc_auc_score(y_test, baseline_proba)
baseline_brier = brier_score_loss(y_test, baseline_proba)

# KS statistic
from scipy.stats import ks_2samp
ks_stat_baseline, _ = ks_2samp(
    baseline_proba[y_test == 0], baseline_proba[y_test == 1]
)

print(f"  Baseline ROC-AUC: {baseline_auc:.4f}")

baseline_metric = {
    "model": "Vanilla Logistic Regression",
    "metric_name": "ROC-AUC",
    "value": round(float(baseline_auc), 6),
    "KS": round(float(ks_stat_baseline), 6),
    "Brier": round(float(baseline_brier), 6),
}

with open(os.path.join(OUTPUTS_DIR, "baseline_metric.json"), "w") as f:
    json.dump(baseline_metric, f, indent=2)
print(f"  ✅ Saved outputs/baseline_metric.json")

# ---------------------------------------------------------------------------
# 5. Primary model — LightGBM
# ---------------------------------------------------------------------------
print("\nTraining primary model (LightGBM) …")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
)

primary_proba = lgb_model.predict_proba(X_test)[:, 1]
primary_auc = roc_auc_score(y_test, primary_proba)
primary_brier = brier_score_loss(y_test, primary_proba)

ks_stat_primary, _ = ks_2samp(
    primary_proba[y_test == 0], primary_proba[y_test == 1]
)

THRESHOLD = 0.72
passed = primary_auc >= THRESHOLD

print(f"  Primary ROC-AUC: {primary_auc:.4f}  (threshold {THRESHOLD}) — {'PASS ✅' if passed else 'FAIL ❌'}")

primary_metric = {
    "model": "LightGBM",
    "metric_name": "ROC-AUC",
    "value": round(float(primary_auc), 6),
    "passed": passed,
    "threshold": THRESHOLD,
    "baseline_logreg_auc": round(float(baseline_auc), 6),
    "KS": round(float(ks_stat_primary), 6),
    "Brier": round(float(primary_brier), 6),
}

with open(os.path.join(OUTPUTS_DIR, "primary_metric.json"), "w") as f:
    json.dump(primary_metric, f, indent=2)
print(f"  ✅ Saved outputs/primary_metric.json")

# ---------------------------------------------------------------------------
# 6. Milestone manifest
# ---------------------------------------------------------------------------
import datetime

manifest = {
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "data_rows_after_filter": int(df.shape[0]),
    "features_used": int(len(numeric_features)),
    "test_set_size": int(len(X_test)),
    "outputs": {
        "baseline_metric": "outputs/baseline_metric.json",
        "primary_metric": "outputs/primary_metric.json",
        "milestone_manifest": "outputs/milestone_manifest.json",
    },
    "baseline_roc_auc": round(float(baseline_auc), 6),
    "primary_roc_auc": round(float(primary_auc), 6),
    "primary_passed": passed,
}

with open(os.path.join(OUTPUTS_DIR, "milestone_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"  ✅ Saved outputs/milestone_manifest.json")

print("\nDone.")
