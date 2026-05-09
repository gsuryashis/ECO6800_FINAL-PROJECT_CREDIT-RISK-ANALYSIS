"""
main.py — ECO 6800 Credit Risk Analysis Pipeline
Run: python main.py

Writes four files to outputs/:
  baseline_metric.json
  primary_metric.json
  scorecard_comparison.csv
  milestone_manifest.json
"""

import json
import os
import warnings
import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Utils — force pure Python types (critical for JSON serialisation)
# ---------------------------------------------------------------------------
def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH          = os.path.join("data", "raw", "lc_loan.csv")
FALLBACK_DATA_PATH = os.path.join("data", "fallback", "lc_loan.csv")
OUTPUTS_DIR        = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── success criteria ──
THRESHOLD      = 0.72
LIFT_THRESHOLD = 0.04

def resolve_data_path():
    if os.path.exists(DATA_PATH):
        return DATA_PATH, "raw"
    if os.path.exists(FALLBACK_DATA_PATH):
        return FALLBACK_DATA_PATH, "fallback"
    raise FileNotFoundError(
        f"No dataset found.\n"
        f"  Expected raw      : {DATA_PATH}\n"
        f"  Expected fallback : {FALLBACK_DATA_PATH}\n"
        "Place lc_loan.csv in data/raw/ or data/fallback/ and retry."
    )

# ---------------------------------------------------------------------------
# 1. Load & build target
# ---------------------------------------------------------------------------
print("Loading data …")
resolved_data_path, data_mode = resolve_data_path()
print(f"  Source : {resolved_data_path}  [{data_mode}]")

df = pd.read_csv(resolved_data_path, low_memory=False)
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

keep_statuses = {"Charged Off", "Default", "Late (31-120 days)", "Fully Paid"}
df = df[df["loan_status"].isin(keep_statuses)].copy()

df["defaultstatus"] = (
    df["loan_status"].isin({"Charged Off", "Default", "Late (31-120 days)"})
).astype(int)

print(f"  After filtering : {df.shape[0]:,} rows")
print(f"  Bad rate        : {df['defaultstatus'].mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# 2. Feature selection — numeric application-time features only
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

X = df[numeric_features].fillna(df[numeric_features].median(numeric_only=True))
y = df["defaultstatus"]
print(f"  Features used   : {len(numeric_features)}")

# ---------------------------------------------------------------------------
# 3. 80/20 stratified split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# ---------------------------------------------------------------------------
# 4. Baseline — vanilla logistic regression
# ---------------------------------------------------------------------------
print("\nTraining baseline (Logistic Regression) …")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_sc, y_train)

baseline_proba  = lr.predict_proba(X_test_sc)[:, 1]
baseline_auc    = float(roc_auc_score(y_test, baseline_proba))
baseline_brier  = float(brier_score_loss(y_test, baseline_proba))
ks_stat_baseline = float(ks_2samp(
    baseline_proba[y_test == 0],
    baseline_proba[y_test == 1]
)[0])

print(f"  Baseline ROC-AUC : {baseline_auc:.4f}")

baseline_metric = {
    "model"       : "Logistic Regression",
    "metric_name" : "ROC-AUC",
    "value"       : round(baseline_auc, 6),
    "KS"          : round(ks_stat_baseline, 6),
    "Brier"       : round(baseline_brier, 6),
}

with open(os.path.join(OUTPUTS_DIR, "baseline_metric.json"), "w") as f:
    json.dump(to_python_types(baseline_metric), f, indent=2)
print("  ✅ Saved baseline_metric.json")

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
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=-1),
    ],
)

primary_proba   = lgb_model.predict_proba(X_test)[:, 1]
primary_auc     = float(roc_auc_score(y_test, primary_proba))
primary_brier   = float(brier_score_loss(y_test, primary_proba))
ks_stat_primary = float(ks_2samp(
    primary_proba[y_test == 0],
    primary_proba[y_test == 1]
)[0])

lift   = float(primary_auc - baseline_auc)
passed = bool(primary_auc >= THRESHOLD and lift >= LIFT_THRESHOLD)

print(f"  Primary ROC-AUC : {primary_auc:.4f}  (threshold {THRESHOLD}) — {'PASS ✅' if passed else 'FAIL ❌'}")

primary_metric = {
    "model"               : "LightGBM",
    "metric_name"         : "ROC-AUC",
    "value"               : round(primary_auc, 6),
    "passed"              : passed,
    "threshold"           : float(THRESHOLD),
    "lift_over_baseline"  : round(lift, 6),
    "lift_threshold"      : float(LIFT_THRESHOLD),
    "baseline_logreg_auc" : round(baseline_auc, 6),
    "KS"                  : round(ks_stat_primary, 6),
    "Brier"               : round(primary_brier, 6),
}

with open(os.path.join(OUTPUTS_DIR, "primary_metric.json"), "w") as f:
    json.dump(to_python_types(primary_metric), f, indent=2)
print("  ✅ Saved primary_metric.json")

# ---------------------------------------------------------------------------
# 6. SHAP scorecard comparison
# ---------------------------------------------------------------------------
print("\nBuilding SHAP scorecard comparison …")

sample_size = min(5000, len(X_test))
X_sample    = X_test.sample(n=sample_size, random_state=42)

explainer = shap.TreeExplainer(lgb_model)

try:
    shap_values = explainer.shap_values(X_sample)

    # FIX 1 — handle list (old shap) or 3-D array (new shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

except Exception as e:
    print(f"  ⚠️  SHAP failed ({e}), using zeros")
    shap_values = np.zeros_like(X_sample.values)

# FIX 2 — use X_sample.columns (actual order) not numeric_features
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_sample.columns,
    name="SHAP_Mean_Abs_Importance",
)

shap_score_range = (mean_abs_shap / mean_abs_shap.max() * 100).fillna(0)

comparison = pd.DataFrame({
    "Feature"         : mean_abs_shap.index,
    "SHAP_Importance" : mean_abs_shap.values,
    "SHAP_Score"      : shap_score_range.values,
}).sort_values("SHAP_Importance", ascending=False)

comparison.to_csv(os.path.join(OUTPUTS_DIR, "scorecard_comparison.csv"), index=False)
print(f"  ✅ Saved scorecard_comparison.csv  ({len(comparison)} features)")

# ---------------------------------------------------------------------------
# 7. Milestone manifest
# ---------------------------------------------------------------------------
print("\nWriting milestone manifest …")

manifest = {
    "charter_locked": True,
    "sources": [
        {
            "name": "LendingClub Issued Loans",
            "status": data_mode,
            "probe_artifact": resolved_data_path,
        }
    ],
    "baseline_ready": os.path.exists(os.path.join(OUTPUTS_DIR, "baseline_metric.json")),
    "primary_metric_schema_ready": all(
        k in primary_metric for k in ("metric_name", "value", "threshold", "passed")
    ),
    "run_command": "uv run main.py",
    # FIX 3 — utcnow() deprecated in Python 3.12+
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "data_mode_used": data_mode,
    "data_source_path": resolved_data_path,
    "source_access_proof": {
        "expected_raw_path": DATA_PATH,
        "fallback_sample_path": FALLBACK_DATA_PATH,
        "probe_script": "scripts/probe_data.py",
        "source_reference_file": "data/raw_data_link.md",
        "external_sources": [
            "https://drive.google.com/drive/u/2/folders/1V1iKS0rcghr6K5paWJDK38X3K8lmDn5P",
            "https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans",
        ],
    },
    "data_rows_after_filter": int(df.shape[0]),
    "features_used": int(len(numeric_features)),
    "test_set_size": int(len(X_test)),
    "outputs": {
        "baseline_metric": "outputs/baseline_metric.json",
        "primary_metric": "outputs/primary_metric.json",
        "scorecard_comparison": "outputs/scorecard_comparison.csv",
        "milestone_manifest": "outputs/milestone_manifest.json",
    },
    "baseline_roc_auc": round(baseline_auc, 6),
    "primary_roc_auc": round(primary_auc, 6),
    "primary_passed": passed,
    "gini_coefficient": round(float(2 * primary_auc - 1), 6),
    "hypothesis_result": (
        f"{'SUPPORTED' if passed else 'NOT SUPPORTED'}: "
        f"LightGBM AUC = {primary_auc:.4f}, "
        f"LR baseline AUC = {baseline_auc:.4f}, "
        f"lift = {lift:.4f} (required >= {LIFT_THRESHOLD}), "
        f"AUC threshold passed = {bool(primary_auc >= THRESHOLD)}, "
        f"lift threshold passed = {bool(lift >= LIFT_THRESHOLD)}"
    ),
}

with open(os.path.join(OUTPUTS_DIR, "milestone_manifest.json"), "w") as f:
    json.dump(to_python_types(manifest), f, indent=2)
print("  ✅ Saved milestone_manifest.json")

print("\nDone.")
