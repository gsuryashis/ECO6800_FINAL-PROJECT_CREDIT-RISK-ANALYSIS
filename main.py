"""
main.py — ECO 6810 Credit Risk Analysis Pipeline
Run: uv run main.py

Writes four files to outputs/:
  baseline_metric.json     — vanilla logistic regression ROC-AUC
  primary_metric.json      — LightGBM ROC-AUC (primary success criterion)
  scorecard_comparison.csv — WoE scorecard point ranges vs. SHAP-based LightGBM importances
  milestone_manifest.json  — summary of all outputs produced
"""

import datetime
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH          = os.path.join("data", "raw", "lc_loan.csv")
FALLBACK_DATA_PATH = os.path.join("data", "fallback", "lc_loan.csv")
OUTPUTS_DIR        = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def resolve_data_path():
    if os.path.exists(DATA_PATH):
        return DATA_PATH, "raw"
    if os.path.exists(FALLBACK_DATA_PATH):
        return FALLBACK_DATA_PATH, "fallback"
    raise FileNotFoundError(
        f"❌ Data file not found at {DATA_PATH}.\n"
        f"   Fallback also missing at {FALLBACK_DATA_PATH}.\n"
        "   Download lc_loan.csv from Google Drive / Kaggle and place it in data/raw/,\n"
        "   or restore the committed fallback file.\n"
        "   See README.md for full instructions."
    )


# ---------------------------------------------------------------------------
# Helper — guarantee every value written to JSON is a plain Python scalar.
# numpy bool_, float32, int64, etc. are NOT JSON-serialisable in Python 3.14.
# ---------------------------------------------------------------------------
def to_py(val):
    """Recursively convert numpy scalars → native Python types."""
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ---------------------------------------------------------------------------
# 1. Load & build target
# ---------------------------------------------------------------------------
print("Loading data …")
resolved_data_path, data_mode = resolve_data_path()
if data_mode == "fallback":
    print(f"  ⚠️  Raw data missing; using fallback sample at {resolved_data_path}")
else:
    print(f"  Using raw dataset at {resolved_data_path}")

df = pd.read_csv(resolved_data_path, low_memory=False)
print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Keep only terminal loan statuses (exclude right-censored)
keep_statuses = {"Charged Off", "Default", "Late (31-120 days)", "Fully Paid"}
df = df[df["loan_status"].isin(keep_statuses)].copy()

# Binary target: 1 = bad (default), 0 = good (fully paid)
df["defaultstatus"] = (
    df["loan_status"].isin({"Charged Off", "Default", "Late (31-120 days)"})
).astype(int)

print(f"  After filtering: {df.shape[0]:,} rows")
print(f"  Bad rate: {df['defaultstatus'].mean() * 100:.2f}%")

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
X_test_sc  = scaler.transform(X_test)

baseline_lr = LogisticRegression(max_iter=500, random_state=42)
baseline_lr.fit(X_train_sc, y_train)

baseline_proba = baseline_lr.predict_proba(X_test_sc)[:, 1]
baseline_auc   = float(roc_auc_score(y_test, baseline_proba))   # ← plain float
baseline_brier = float(brier_score_loss(y_test, baseline_proba))

ks_stat_baseline, _ = ks_2samp(
    baseline_proba[y_test == 0], baseline_proba[y_test == 1]
)
ks_stat_baseline = float(ks_stat_baseline)

print(f"  Baseline ROC-AUC: {baseline_auc:.4f}")

baseline_metric = {
    "model":       "Vanilla Logistic Regression",
    "metric_name": "ROC-AUC",
    "value":       round(baseline_auc, 6),
    "KS":          round(ks_stat_baseline, 6),
    "Brier":       round(baseline_brier, 6),
}

with open(os.path.join(OUTPUTS_DIR, "baseline_metric.json"), "w") as f:
    json.dump(baseline_metric, f, indent=2)
print("  ✅ Saved outputs/baseline_metric.json")

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
primary_auc   = float(roc_auc_score(y_test, primary_proba))     # ← plain float
primary_brier = float(brier_score_loss(y_test, primary_proba))

ks_stat_primary, _ = ks_2samp(
    primary_proba[y_test == 0], primary_proba[y_test == 1]
)
ks_stat_primary = float(ks_stat_primary)

THRESHOLD      = 0.72
LIFT_THRESHOLD = 0.04
lift = primary_auc - baseline_auc

# FIX: all comparisons are between plain Python floats now, so `passed` is a
# native bool — safe to serialise with json.dump on any Python version.
passed = bool(primary_auc >= THRESHOLD) and bool(lift >= LIFT_THRESHOLD)

print(
    f"  Primary ROC-AUC: {primary_auc:.4f}  "
    f"(threshold {THRESHOLD}) — {'PASS ✅' if passed else 'FAIL ❌'}"
)

primary_metric = {
    "model":              "LightGBM",
    "metric_name":        "ROC-AUC",
    "value":              round(primary_auc, 6),
    "passed":             passed,                        # native bool ✓
    "threshold":          float(THRESHOLD),
    "lift_over_baseline": round(lift, 6),
    "lift_threshold":     float(LIFT_THRESHOLD),
    "baseline_logreg_auc": round(baseline_auc, 6),
    "KS":                 round(ks_stat_primary, 6),
    "Brier":              round(primary_brier, 6),
}

with open(os.path.join(OUTPUTS_DIR, "primary_metric.json"), "w") as f:
    json.dump(primary_metric, f, indent=2)
print("  ✅ Saved outputs/primary_metric.json")

# ---------------------------------------------------------------------------
# 6. Scorecard comparison — WoE scorecard vs. SHAP-based LightGBM scores
# ---------------------------------------------------------------------------
print("\nBuilding scorecard comparison …")

sample_size = min(5000, len(X_test))
X_sample    = X_test.sample(n=sample_size, random_state=42)

explainer   = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_sample)
if isinstance(shap_values, list):        # older shap versions return a list
    shap_values = shap_values[1]

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=numeric_features,
    name="SHAP_Mean_Abs_Importance",
)

shap_max        = mean_abs_shap.max()
shap_score_range = (mean_abs_shap / shap_max * 100).round(1).rename("SHAP_Score_Range")

woe_scorecard_path = os.path.join(OUTPUTS_DIR, "tables", "final_scorecard.csv")
if os.path.exists(woe_scorecard_path):
    woe_sc      = pd.read_csv(woe_scorecard_path)
    woe_summary = woe_sc.groupby("Feature")["Points"].agg(
        WoE_Min_Points="min",
        WoE_Max_Points="max",
        WoE_Point_Range=lambda x: x.max() - x.min(),
    ).reset_index()
else:
    woe_summary = pd.DataFrame(
        columns=["Feature", "WoE_Min_Points", "WoE_Max_Points", "WoE_Point_Range"]
    )

shap_df = pd.DataFrame({
    "Feature":                  mean_abs_shap.index,
    "SHAP_Mean_Abs_Importance": mean_abs_shap.round(6).values,
    "SHAP_Score_Range":         shap_score_range.values,
})

comparison = woe_summary.merge(shap_df, on="Feature", how="outer").sort_values(
    "SHAP_Mean_Abs_Importance", ascending=False, na_position="last"
)

comparison.to_csv(os.path.join(OUTPUTS_DIR, "scorecard_comparison.csv"), index=False)
print(f"  ✅ Saved outputs/scorecard_comparison.csv  ({len(comparison)} features)")

# ---------------------------------------------------------------------------
# 7. Milestone manifest
# ---------------------------------------------------------------------------
manifest = {
    "generated_at":    datetime.datetime.utcnow().isoformat() + "Z",
    "data_mode_used":  data_mode,
    "data_source_path": resolved_data_path,
    "source_access_proof": {
        "expected_raw_path":    DATA_PATH,
        "fallback_sample_path": FALLBACK_DATA_PATH,
        "probe_script":         "scripts/probe_data.py",
        "source_reference_file": "data/raw_data_link.md",
        "external_sources": [
            "https://drive.google.com/drive/u/2/folders/1V1iKS0rcghr6K5paWJDK38X3K8lmDn5P",
            "https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans",
        ],
    },
    "data_rows_after_filter": int(df.shape[0]),
    "features_used":          int(len(numeric_features)),
    "test_set_size":          int(len(X_test)),
    "outputs": {
        "baseline_metric":   "outputs/baseline_metric.json",
        "primary_metric":    "outputs/primary_metric.json",
        "milestone_manifest": "outputs/milestone_manifest.json",
        "scorecard_comparison": "outputs/scorecard_comparison.csv",
    },
    "baseline_roc_auc":   round(baseline_auc, 6),
    "primary_roc_auc":    round(primary_auc, 6),
    "primary_passed":     passed,                           # native bool ✓
    "gini_coefficient":   round(float(2 * primary_auc - 1), 6),
    "hypothesis_result": (
        f"{'SUPPORTED' if passed else 'NOT SUPPORTED'}: "
        f"LightGBM AUC = {primary_auc:.4f}, LR baseline AUC = {baseline_auc:.4f}, "
        f"lift = {lift:.4f} (required >= {LIFT_THRESHOLD}), "
        f"AUC threshold passed = {bool(primary_auc >= THRESHOLD)}, "
        f"lift threshold passed = {bool(lift >= LIFT_THRESHOLD)}"
    ),
}

with open(os.path.join(OUTPUTS_DIR, "milestone_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("  ✅ Saved outputs/milestone_manifest.json")

print("\nDone. ✅")
