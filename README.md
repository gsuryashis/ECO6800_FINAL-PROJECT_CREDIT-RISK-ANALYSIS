# Credit Risk Scorecard Development

## Quick start

### 1 — Data file behavior

If you are on Windows and need to install `uv`, run:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:PATH += ";$env:USERPROFILE\.local\bin"
```

The PATH line applies to the current session; then restart your terminal.

`uv run main.py` and `uv run scripts/probe_data.py` now work on a clean clone using the committed fallback sample:

```
data/fallback/lc_loan.csv
```

> Note: `data/fallback/lc_loan.csv` is a small synthetic sample dataset included only for reproducibility/testing on clean machines. It is not the full LendingClub source dataset used for final project analysis.

To run with the full LendingClub dataset instead, download `lc_loan.csv` from one of the sources below and put it at exactly this path inside the repo:

```
data/raw/lc_loan.csv
```

Sources:

- Google Drive: <https://drive.google.com/drive/u/2/folders/1V1iKS0rcghr6K5paWJDK38X3K8lmDn5P>
- Kaggle: <https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans>

### 2 — Install dependencies (optional with uv)

If you use uv, dependencies are declared in `pyproject.toml`, so `uv run main.py` works directly on a fresh clone.
`requirements.txt` now delegates to the project metadata (`.`), so `pyproject.toml` is the single dependency source of truth.

```bash
pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -r requirements.txt
```

### 3 — (Optional) probe the data file

```bash
uv run scripts/probe_data.py
```

This verifies the file exists and contains the expected columns.

### 4 — Run the full pipeline

```bash
uv run main.py
```

**Expected outputs** (written to `outputs/`):

| File | Contents |
|---|---|
| `outputs/baseline_metric.json` | Vanilla logistic regression ROC-AUC, KS, and Brier score |
| `outputs/primary_metric.json` | LightGBM ROC-AUC + pass/fail against 0.72 threshold, KS, Brier |
| `outputs/milestone_manifest.json` | Run summary with Gini coefficient and links to all output files |
| `outputs/scorecard_comparison.csv` | WoE scorecard point ranges vs. SHAP-based LightGBM feature importances |

Pre-computed versions of these files are committed to the repo and will be overwritten by a fresh run.

### Detailed pipeline (individual scripts)

Run scripts in order if you want the full WoE scorecard pipeline:

1. `src/pre_processing.py`
2. `src/EDA.py`
3. `src/woe2.py`
4. `src/model.py`
5. `src/score_card.py`
6. `src/score_new_loans.py`

This repository contains a proposal for a credit risk scorecard project designed to predict the probability that a borrower will default on a loan. The project is framed as an interpretable, end-to-end scorecard pipeline rather than a black-box prediction exercise, making it suitable for academic review and discussion.

## Project Overview

The proposed study uses historical loan-level data to build a binary credit risk model that classifies borrowers into default and non-default groups. The core outcome variable is `defaultstatus`, where loans marked as `Charged Off` or `Default` are treated as bad loans and all other eligible observations are treated as good loans.

The workflow described in the project materials follows a standard scorecard-development sequence: data preprocessing, exploratory data analysis, feature engineering, Weight of Evidence (WoE) transformation, Information Value (IV) based variable selection, logistic regression modeling, score scaling, and validation on unseen data.

## Research Objective

The main objective is to develop an interpretable scorecard that can estimate default risk for new loan applicants using borrower, loan, and credit-profile characteristics available in the dataset.

More specifically, the project intends to:

- Define a transparent default target suitable for credit risk modeling.
- Clean and preprocess a large loan-level dataset with substantial missingness across several variables.
- Identify the most predictive risk factors using WoE and IV methods.
- Train a logistic regression model that can be translated into a business-friendly point-based scorecard.
- Provide a reproducible modeling pipeline that can later be applied to new borrower data.

## Dataset

The proposal is based on a large historical loan dataset containing 887,379 observations and roughly 72 to 74 raw variables, depending on the preprocessing stage being referenced in the working notes.

The available variables include borrower demographics and employment indicators, loan contract characteristics, repayment information, and credit bureau style variables such as debt-to-income ratio, delinquencies, revolving utilization, total accounts, and inquiry measures.

### Data files

The project data files are available here:

- Google Drive folder: <https://drive.google.com/drive/u/2/folders/1V1iKS0rcghr6K5paWJDK38X3K8lmDn5P>

If the repository is shared publicly, data should preferably remain external to the repo and be accessed from the Drive folder because the files are likely too large for version control and may require controlled sharing permissions.

## Target Definition

The proposed dependent variable is `defaultstatus`, a binary flag constructed from the `loanstatus` field.

- `defaultstatus = 1`: loans classified as `Charged Off` or `Default`.
- `defaultstatus = 0`: loans classified as non-default / good loans after the chosen filtering and preprocessing rules.

This target definition is appropriate for scorecard modeling because it focuses on clear adverse credit outcomes rather than temporary delinquency states that may later cure.

## Proposed Methodology

### 1. Data preprocessing

The first step is to clean the raw loan file, create the target variable, assess missing values, and remove variables that are either unusable, excessively sparse, or clear sources of target leakage.

The project notes indicate that variables with very high missingness, especially joint-application fields and some advanced credit metrics, were candidates for removal. The proposal also distinguishes between predictive features and variables that reflect post-default outcomes, such as recoveries and collection-related measures, which should not be used when building a proper ex-ante risk model.

### 2. Exploratory data analysis

EDA is intended to identify early risk patterns, compare default rates across categorical borrower groups, examine the distribution of numerical variables, and detect multicollinearity among highly related loan variables.

The materials suggest that loan grade, subgrade, interest rate, revolving utilization, home ownership, and loan purpose are important dimensions to study during this phase.

### 3. Feature engineering and selection

The core feature-engineering strategy is based on Weight of Evidence transformation and Information Value ranking. This is a standard scorecard approach because WoE helps create a more stable and interpretable relationship between predictors and the log-odds of default, while IV provides a transparent way to assess predictive strength.
The project proposes retaining variables with acceptable predictive contribution and excluding variables that are either too weak or suspiciously strong in ways that may indicate leakage.

### 4. Model development

The planned predictive model is logistic regression. This choice is aligned with scorecard construction because logistic regression is interpretable, well suited for binary default prediction, and can be directly converted into a points-based scoring framework.

A training-test split is proposed so that model performance can be evaluated on hold-out data rather than judged only on in-sample fit.

### 5. Scorecard construction

After model estimation, the logistic regression output is intended to be scaled into a practical scorecard where higher scores indicate lower credit risk. The proposal describes a point-based system calibrated around a reference score and odds relationship so that the final product is easier to interpret than raw model coefficients.

This makes the project relevant not only as a predictive modeling exercise but also as an example of translating statistical output into a decision-support tool.

### 6. Validation and deployment logic

The final stage is to test whether the same preprocessing rules, WoE mappings, and model artifacts can be consistently applied to new data. In a full implementation, this would allow the repository to support repeatable scoring of unseen borrower observations.

## Repository Structure

A simple repository structure for this project could look like this:

```text
credit-risk-scorecard/
├── data/
│   └── raw_data_link.md
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_woe_iv.ipynb
│   ├── 04_model_scorecard.ipynb
│   └── 05_validation.ipynb
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── scorecard_model.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── artifacts/
├── requirements.txt
└── README.md
```

This layout keeps the workflow modular and makes it easy for an instructor to inspect the logic of the project from raw data handling through final model construction.

## Planned Deliverables

This proposal is intended to lead to the following project outputs:

- A cleaned analytical dataset for scorecard modeling.
- Exploratory analysis of borrower and loan characteristics associated with default risk.
- WoE binning tables and IV-based variable selection outputs.
- A logistic regression based credit risk scorecard.
- Reusable scoring artifacts that can be applied to future datasets.

## Why This Project Matters

Credit risk modeling is central to lending decisions because it helps institutions quantify expected borrower risk, improve screening, and standardize approval logic. A scorecard framework is especially useful in an academic project because it combines statistical rigor with transparency and interpretability.

The proposed design also provides a strong applied example of how data preprocessing, careful target definition, leakage control, and model validation matter in real-world predictive analytics.


## Results

| Metric | Vanilla LR Baseline | LightGBM (Primary) |
|---|---|---|
| ROC-AUC | 0.6920 | **0.7742** |
| Gini Coefficient | 0.3840 | **0.5484** |
| KS Statistic | 0.3218 | 0.4283 |
| Brier Score | 0.1486 | 0.1219 |

- **Hypothesis supported:** LightGBM AUC (0.7742) exceeds vanilla LR AUC (0.6920) by 0.0822, well above the 0.04 threshold.
- Final WoE scorecard available in `outputs/tables/final_scorecard.csv`
- Model comparison available in `outputs/scorecard_comparison.csv`
