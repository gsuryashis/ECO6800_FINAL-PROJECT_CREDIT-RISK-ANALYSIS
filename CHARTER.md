# Project Charter — ECO 6810 Final Project

## Header

| Field | Value |
|---|---|
| Team members | Suryashis, Vignesh, Vishvas |
| Project type | Predictive |
| Estimated hours per person | 60 hours |
| Charter version | v3 |
| Date | 2026-05-02 |

---

## 1. Problem and stakeholder

The Chief Risk Officer (CRO) of a consumer lending bank needs to estimate retail borrower default probability using fully transparent, regulatory-compliant models (ECOA/FCRA standards). While traditional Weight of Evidence (WoE) scorecards are interpretable, they often sacrifice the predictive power of modern algorithms.

This project develops two interpretable scorecards — a traditional WoE-binned Logistic Regression Scorecard and a LightGBM scorecard with SHAP-derived feature scores — to give the CRO a quantitative basis for evaluating whether a modern gradient-boosted model can replace a legacy logistic framework in regulated credit underwriting.

---

## 2. Exact Prediction Target

- **Name:** `defaultstatus` (Probability of Default)
- **Source file:** `data/raw/lc_loan.csv` (LendingClub Issued Loans, Kaggle)
- **Source column:** `loan_status`
- **Construction:**
  - **1 (Default/Bad):** Rows where `loan_status` ∈ {`Charged Off`, `Default`, `Late (31-120 days)`}
  - **0 (Non-Default/Good):** Rows where `loan_status` = `Fully Paid`
  - **Excluded rows:** `loan_status` ∈ {`Current`, `In Grace Period`} — these are right-censored and excluded entirely.
- **Unit:** Binary (0/1) — no probability labels are used; class membership is deterministic from `loan_status`.
- **Population:** ~887k total observations before outcome filtering; approximately 200k–250k rows expected after exclusions.

---

## 3. Held-out metric for success

The model will be evaluated on a **20% stratified held-out test set**. Success is defined by a single primary metric:

> **Held-out ROC-AUC ≥ 0.72** for the LightGBM model, versus a vanilla logistic regression baseline ROC-AUC of approximately **0.68** (established on the same held-out set).

This constitutes a minimum improvement of **0.04 ROC-AUC points** over baseline, meeting industry standards for meaningful discrimination lift. Secondary diagnostics (KS statistic, Brier Score) will be reported but are not gating criteria.

---

## 4. Baseline to beat

1. **Majority-Class Baseline:** Predicting the modal outcome (Fully Paid) for all observations.
2. **Simple Rules-Based Heuristic:** A naive policy rule — predict Default = 1 if `grade` ∈ {`E`, `F`, `G`} OR `dti` > 30%.
3. **Vanilla Logistic Regression (primary baseline):** An unbinned, continuous-variable logistic regression fit on the training set. This is the model the primary metric threshold is benchmarked against.

---

## 5. Falsifiable hypothesis

**Hypothesis:** LightGBM will improve held-out ROC-AUC over vanilla logistic regression by at least **0.04** (e.g., ≥ 0.72 vs. ~0.68).

---

## 6. Exact data sources and access status

- **Original Source:** Kaggle — [LendingClub Issued Loans Dataset](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans?resource=download)
- **Mirrored to:** Shared Google Drive — [Drive Link](https://drive.google.com/drive/u/2/folders/1V1iKS0rcghr6K5paWJDK38X3K8lmDn5P)
- **Local probe path:** `data/raw/lc_loan.csv`
- **Clean-machine access:** A Kaggle account or Google Drive access is required. The file is not bundled in the repo. Team members must manually download and place it at `data/raw/lc_loan.csv` before running.
- **Data probe script:** `scripts/probe_data.py` — run via `uv run scripts/probe_data.py`

```python
# scripts/probe_data.py
import pandas as pd
import os

DATA_PATH = os.path.join("data", "raw", "lc_loan.csv")

def probe_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data missing. Place CSV at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, nrows=5)
    required_cols = ["loan_status", "loan_amnt", "dti", "grade"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing expected columns: {missing}")
    print(f"✅ Data loaded. Columns present: {df[required_cols].columns.tolist()}")
    return True

if __name__ == "__main__":
    probe_data()
```

---

## 7. Scope limits

- We model **Probability of Default (PD) only**; we will not estimate Loss Given Default (LGD) or Exposure at Default (EAD).
- We are building an **origination/application scorecard** — all post-approval behavioral variables (e.g., `recoveries`, `total_rec_prncp`, `last_pymnt_amnt`) are strictly dropped to prevent data leakage.
- **No causal claims:** SHAP values measure feature attribution within the model, not causal effects of loan features on default. We make no causal claims.
- **No production deployment:** This project is an academic comparison exercise. Results do not constitute, and should not be construed as, a production lending decision system.
- **No claim that SHAP proves causality:** SHAP is used for interpretability and scorecard construction only.

---

## 8. Risks and fallback

**Risk:** SHAP values from the LightGBM model exhibit non-linearities that cannot be smoothed into a monotonic, compliant scorecard format.

**Fallback:** Enforce monotonic constraints natively via LightGBM's `monotone_constraints` hyperparameter, forcing the algorithm to learn only regulatory-compliant trees before extracting SHAP values.

---

## 9. Reproducibility checklist

- `uv run main.py` runs the pipeline end-to-end.
- Writes `outputs/primary_metric.json`:
  ```json
  {
    "metric_name": "ROC-AUC",
    "value": <num>,
    "passed": true,
    "threshold": 0.72,
    "baseline_logreg_auc": <num>,
    "KS": <num>,
    "Brier": <num>
  }
  ```
  The `"passed"` field is `true` if `value >= 0.72`, `false` otherwise.
- Writes `outputs/scorecard_comparison.csv` showing point allocations for both WoE and SHAP scorecards.
- `README.md` documents setup commands, data placement instructions, and expected outputs.

---

Signed: Suryashis, Vignesh, Vishvas
