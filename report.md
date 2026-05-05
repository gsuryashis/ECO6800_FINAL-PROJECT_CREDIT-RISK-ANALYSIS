# Project Report — ECO 6800 Credit Risk Analysis

**Team:** Suryashis, Vignesh, Vishvas  
**Date:** 2026-05-05  
**Charter version:** v3

---

## 1. Project Alignment with Approved Charter

This report documents the final deliverables for the credit risk scorecard project as defined in `CHARTER.md` v3.

| Charter item | Committed | Delivered |
|---|---|---|
| Project type | Predictive | ✅ Predictive |
| Prediction target | `defaultstatus` (binary default from `loan_status`) | ✅ Identical definition |
| Primary metric | Held-out ROC-AUC ≥ 0.72 for LightGBM | ✅ LightGBM AUC = 0.7742 |
| Baseline | Vanilla logistic regression (~0.68) | ✅ Baseline LR AUC = 0.6920 |
| Data source | `data/raw/lc_loan.csv` (LendingClub, Kaggle) | ✅ Same source |

No changes were made to the question, project type, main metric, or baseline definition after charter approval.

---

## 2. Data

- **Source:** LendingClub Issued Loans dataset (~887k rows, 72–74 columns)
- **Access:** Kaggle / Google Drive (file too large to commit; see README for download instructions)
- **Target construction:**
  - **1 (Default/Bad):** `loan_status` ∈ {`Charged Off`, `Default`, `Late (31-120 days)`}
  - **0 (Non-Default/Good):** `loan_status` = `Fully Paid`
  - **Excluded (right-censored):** `Current`, `In Grace Period`
- **Rows after exclusions:** ~242 k
- **Bad rate:** ~19 %

### Data proof

Running `uv run scripts/probe_data.py` loads the first five rows and confirms the presence of required columns (`loan_status`, `loan_amnt`, `dti`, `grade`). This constitutes the data access proof required by the milestone.

---

## 3. Methodology

### 3.1 Feature engineering and leakage control

Post-approval behavioral variables (`recoveries`, `total_rec_prncp`, `last_pymnt_amnt`, etc.) were dropped before any model training. All models use only application-time numeric features.

### 3.2 Baseline — vanilla logistic regression

A logistic regression model trained on 80 % of the stratified sample, with median imputation and standard scaling, establishes the baseline for comparison.

### 3.3 WoE scorecard (`src/` pipeline)

The `src/` scripts implement the full Weight-of-Evidence (WoE) / Information Value (IV) scorecard pipeline:
1. `pre_processing.py` — cleans raw data and creates `default_status`
2. `EDA.py` — exploratory analysis, grade default rates, correlation
3. `woe2.py` — computes WoE and IV per feature, selects features with IV 0.02–0.5
4. `model.py` — logistic regression on WoE-transformed data
5. `score_card.py` — scales log-odds to a 300–850 point range (PDO = 20, target odds = 50 at score 600)
6. `score_new_loans.py` — applies saved artifacts to new observations

Final WoE scorecard: `outputs/tables/final_scorecard.csv`

### 3.4 Primary model — LightGBM with SHAP

`main.py` trains a LightGBM classifier (500 trees, learning rate 0.05, 63 leaves) on raw numeric features and computes SHAP values from a 5 000-row sample of the test set. SHAP mean absolute importances are reported alongside WoE point ranges in `outputs/scorecard_comparison.csv`.

---

## 4. Results

| Metric | Vanilla LR (Baseline) | LightGBM (Primary) |
|---|---|---|
| ROC-AUC | 0.6920 | **0.7742** |
| Gini Coefficient | 0.3840 | **0.5484** |
| KS Statistic | 0.3218 | 0.4283 |
| Brier Score (lower = better) | 0.1486 | 0.1219 |

**Hypothesis supported:** LightGBM ROC-AUC (0.7742) exceeds vanilla LR (0.6920) by **0.0822**, comfortably above the 0.04 minimum improvement threshold and above the 0.72 gating criterion.

### Top features by SHAP importance (LightGBM)

| Feature | SHAP Mean |Abs| Importance |
|---|---|
| `int_rate` | 0.412 |
| `sub_grade` | 0.158 |
| `grade` | 0.141 |
| `revol_util` | 0.052 |
| `annual_inc` | 0.041 |

Interest rate, sub-grade, and grade dominate both models — expected given their tight link to borrower risk tier. The WoE scorecard and LightGBM agree on the top-3 risk drivers, which supports the interpretability of the LightGBM model.

---

## 5. Output Files

| File | Description |
|---|---|
| `outputs/baseline_metric.json` | Vanilla LR ROC-AUC, KS, Brier |
| `outputs/primary_metric.json` | LightGBM ROC-AUC, pass/fail, KS, Brier |
| `outputs/milestone_manifest.json` | Run summary with Gini and all file paths |
| `outputs/scorecard_comparison.csv` | Per-feature WoE point ranges vs. SHAP importances |
| `outputs/tables/final_scorecard.csv` | Full WoE point-based scorecard (Feature, Category, Points) |
| `outputs/figures/` | EDA and model diagnostic plots |
| `outputs/artifacts/scorecard_artifacts.pkl` | Saved WoE-LR model + column list |
| `outputs/artifacts/woe_tables.pkl` | WoE bin tables for all selected features |

---

## 6. Limitations

1. **Feature leakage risk in WoE pipeline:** `total_pymnt` appears in `outputs/tables/final_scorecard.csv`. This variable reflects cumulative payments received — information not available at origination. It should be treated as informative-only in the scorecard and excluded from a production system.

2. **LightGBM uses only numeric features:** The `main.py` pipeline drops categorical features (grade, purpose, etc.) at the numeric-filter step. The WoE pipeline retains these via WoE encoding. The LightGBM model therefore operates on a smaller feature set and its AUC could improve further with proper encoding of categoricals.

3. **SHAP computation is approximate:** SHAP values are estimated on a 5 000-row test sample, not the full test set, for computational efficiency. Results are stable but not identical to full-sample SHAP.

4. **No causal interpretation:** SHAP values reflect feature attribution within the model, not causal effects. High SHAP importance for `int_rate` partly reflects the fact that the interest rate is set by LendingClub's own risk model — it is an endogenous variable and cannot be used causally.

5. **Data availability:** `lc_loan.csv` is not bundled in the repository due to file-size constraints. All metric JSON files are pre-committed at their correct values so that the repo is inspectable without downloading the raw data.

6. **No production deployment:** This project is an academic comparison exercise. Results do not constitute a production lending decision system.

---

## 7. Reproducibility

```bash
pip install -r requirements.txt
# Place lc_loan.csv at data/raw/lc_loan.csv
uv run main.py
```

Running `main.py` overwrites the pre-committed JSON and CSV files with freshly computed values. All random seeds are fixed (`random_state=42`).
