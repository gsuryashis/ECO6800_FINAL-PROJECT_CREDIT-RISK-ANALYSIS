# Project Report — ECO 6810 Credit Risk Analysis

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
| Primary metric | Held-out ROC-AUC ≥ 0.72 for LightGBM | ✅ LightGBM AUC = 0.7992 |
| Baseline | Vanilla logistic regression (~0.68) | ✅ Baseline LR AUC = 0.7551 |
| Data source | `data/raw/lc_loan.csv` (preferred) + `data/fallback/lc_loan.csv` (committed reproducibility sample) | ✅ Both paths supported in code and manifest |

No changes were made to the question, project type, main metric, or baseline definition after charter approval.

---

## 2. Data

- **Source:** LendingClub Issued Loans dataset (full source external) + committed fallback sample in `data/fallback/lc_loan.csv`
- **Access:** Kaggle / Google Drive for full dataset; fallback sample is bundled for clean-machine reproducibility
- **Target construction:**
  - **1 (Default/Bad):** `loan_status` ∈ {`Charged Off`, `Default`, `Late (31-120 days)`}
  - **0 (Non-Default/Good):** `loan_status` = `Fully Paid`
  - **Excluded (right-censored):** `Current`, `In Grace Period`
- **Rows after exclusions (committed reproducible run):** 1,200
- **Bad rate (committed reproducible run):** 54.42%

### Data proof

Running `uv run scripts/probe_data.py` loads the first five rows and confirms the presence of required columns (`loan_status`, `loan_amnt`, `dti`, `grade`) from either `data/raw/lc_loan.csv` or `data/fallback/lc_loan.csv`. This is the source-access proof used by the milestone.

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
| ROC-AUC | 0.7551 | **0.7992** |
| Gini Coefficient | 0.5102 | **0.5984** |
| KS Statistic | 0.4470 | 0.5445 |
| Brier Score (lower = better) | 0.1978 | 0.1731 |

**Hypothesis supported:** LightGBM ROC-AUC (0.7992) exceeds vanilla LR (0.7551) by **0.0441**, above the 0.04 minimum improvement threshold and above the 0.72 gating criterion.

### Top features by SHAP importance (LightGBM)

| Feature | SHAP Mean |Abs| Importance |
|---|---|
| `delinq_2yrs` | 1.059 |
| `loan_amnt` | 0.290 |
| `dti` | 0.237 |
| `annual_inc` | 0.179 |
| `int_rate` | 0.176 |

For the committed fallback run, delinquency history (`delinq_2yrs`) and core affordability/loan-size variables dominate SHAP importance. Feature importance should be re-checked on the full raw dataset when `data/raw/lc_loan.csv` is available.

### Economics interpretation and decision impact

The primary model clears the charter threshold (ROC-AUC ≥ 0.72) and beats the baseline by **0.0441**. In underwriting terms, this means the LightGBM model ranks higher-risk applicants above lower-risk ones more reliably than a vanilla logistic regression. For a Chief Risk Officer, that implies **fewer defaults at a fixed approval rate**, or **more approvals at a fixed default rate**, without changing the underwriting policy objective. Figure `outputs/figures/auc_comparison.png` shows the improvement relative to the threshold.

The effect size is modest but policy-relevant: a +0.044 AUC lift is typically considered a meaningful discrimination gain for retail credit underwriting, especially when the baseline is already in the mid‑0.70s. This is consistent with the charter’s hypothesis that a modern, interpretable boosting model can outperform a legacy scorecard while remaining auditable via SHAP summaries.

### Evidence map (claims → files)

| Claim | Evidence |
|---|---|
| LightGBM meets ROC-AUC ≥ 0.72 and exceeds the baseline by ≥ 0.04 | `outputs/primary_metric.json`, `outputs/baseline_metric.json`, `outputs/figures/auc_comparison.png` |
| Key predictive drivers are delinquency history and affordability metrics | `outputs/scorecard_comparison.csv`, `outputs/tables/shap_top_features.csv`, `outputs/figures/shap_top_features.png` |
| Risk varies systematically by grade (descriptive) | `outputs/tables/grade_default_rates.csv`, `outputs/figures/default_rate_by_grade.png` |
| WoE/IV variable selection evidence exists for the scorecard pipeline | `outputs/tables/iv_summary.csv`, `outputs/tables/woe_bins.csv` |
| Scorecard points vs SHAP importance are comparable | `outputs/scorecard_comparison.csv`, `outputs/tables/final_scorecard.csv` |

### Descriptive vs predictive vs causal claims

- **Descriptive:** Grade default rates (`outputs/tables/grade_default_rates.csv`) describe sample patterns only.
- **Predictive:** ROC-AUC results and SHAP importance are predictive performance and model attribution claims.
- **Causal:** **No causal claims are made.** Interest rate and grade are endogenous to lender policy and should not be interpreted as causal drivers of default.

---

## 5. Output Files

| File | Description |
|---|---|
| `outputs/baseline_metric.json` | Vanilla LR ROC-AUC, KS, Brier |
| `outputs/primary_metric.json` | LightGBM ROC-AUC, pass/fail, KS, Brier |
| `outputs/milestone_manifest.json` | Run summary with Gini and all file paths |
| `outputs/scorecard_comparison.csv` | Per-feature WoE point ranges vs. SHAP importances |
| `outputs/tables/shap_top_features.csv` | Top SHAP features (mean |SHAP| importance) |
| `outputs/tables/grade_default_rates.csv` | Default rates by grade (sample run) |
| `outputs/tables/iv_summary.csv` | Information Value (IV) totals by feature |
| `outputs/tables/woe_bins.csv` | WoE bin-level evidence for scorecard variables |
| `outputs/tables/final_scorecard.csv` | Full WoE point-based scorecard (Feature, Category, Points) |
| `outputs/figures/` | AUC comparison, grade default rates, SHAP feature chart |
| `outputs/artifacts/scorecard_artifacts.pkl` | Saved WoE-LR model + column list |
| `outputs/artifacts/woe_tables.pkl` | WoE bin tables for all selected features |

---

## 6. Limitations

1. **Feature leakage risk in WoE pipeline:** `total_pymnt` appears in `outputs/tables/final_scorecard.csv`. This variable reflects cumulative payments received — information not available at origination. It should be treated as informative-only in the scorecard and excluded from a production system.

2. **LightGBM uses only numeric features:** The `main.py` pipeline drops categorical features (grade, purpose, etc.) at the numeric-filter step. The WoE pipeline retains these via WoE encoding. The LightGBM model therefore operates on a smaller feature set and its AUC could improve further with proper encoding of categoricals.

3. **SHAP computation is approximate:** SHAP values are estimated on a 5 000-row test sample, not the full test set, for computational efficiency. Results are stable but not identical to full-sample SHAP.

4. **No causal interpretation:** SHAP values reflect feature attribution within the model, not causal effects. High SHAP importance for `int_rate` partly reflects the fact that the interest rate is set by LendingClub's own risk model — it is an endogenous variable and cannot be used causally.

5. **Data availability:** The full `data/raw/lc_loan.csv` file is not bundled due to size/privacy constraints. A small permitted fallback sample is committed for reproducibility, and outputs should be interpreted as sample-run artifacts unless regenerated on the full source.

6. **WoE artifacts are precomputed:** `outputs/tables/iv_summary.csv` and `outputs/tables/woe_bins.csv` are exported from the committed WoE artifacts; they should be refreshed when the full raw dataset is available.

7. **No production deployment:** This project is an academic comparison exercise. Results do not constitute a production lending decision system.

---

## 7. What additional evidence would change our mind

1. **Full-data replication:** If the full `data/raw/lc_loan.csv` run reduced the AUC lift below 0.04, the hypothesis would be rejected.
2. **Out-of-time validation:** A holdout from later vintages showing deterioration in LightGBM performance relative to LR would weaken the case for replacement.
3. **Stability across segments:** Evidence that lift only exists in narrow borrower segments would argue against broad deployment.
4. **Fairness diagnostics:** If subgroup performance (e.g., by grade or income bands) shows unacceptable disparities, the model would require redesign.

---

## 8. Reproducibility

```bash
pip install -r requirements.txt
# Optional: place full source at data/raw/lc_loan.csv
uv run scripts/probe_data.py
uv run main.py
```

`main.py` overwrites the pre-committed JSON and CSV files with freshly computed values. If the raw file is missing, the run automatically uses `data/fallback/lc_loan.csv`. All random seeds are fixed (`random_state=42`).
