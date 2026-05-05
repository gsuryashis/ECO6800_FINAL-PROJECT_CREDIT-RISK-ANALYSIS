import pandas as pd
import numpy as np
import pickle

# --- Configuration ---
import os

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
NEW_LOAN_DATA_PATH = os.path.join(DATA_DIR, 'lc_2016_2017(TEST).csv')
MODEL_ARTIFACTS_PATH = 'scorecard_artifacts.pkl'
WOE_TABLES_PATH = 'woe_tables.pkl'
SCORE_OUTPUT_PATH = 'scored_new_loans.csv'

# Constants from original training data for consistent imputation
# (These should match the medians/modes used in your first preprocessing script)
IMPUTATION_VALUES = {
    'revol_util': 56.0,
    'emp_length': 'Unknown',
    'mths_since_last_delinq': 999,
    'mths_since_last_record': 999,
    'mths_since_last_major_derog': 999
    # Add other median/mode values from your original preprocessing here if needed
}


# --- Function Definitions from Original Script ---

def optimal_binning_numerical(df, feature, max_bins=10):
    """
    Creates bins for numerical variables using quantiles.
    """
    if df[feature].nunique() <= max_bins:
        return df[feature]
    try:
        bins = pd.qcut(df[feature], q=max_bins, retbins=True, duplicates='drop')[1]
    except ValueError:
        bins = np.linspace(df[feature].min(), df[feature].max(), max_bins + 1)
    bins = np.unique(bins)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return pd.cut(df[feature], bins=bins)


# --- Main Script ---

# 1. Load New Data and Saved Artifacts
print(">>> STEP 1: LOADING NEW DATA AND SAVED ARTIFACTS...")
try:
    df_new = pd.read_csv(NEW_LOAN_DATA_PATH, low_memory=False)
    print(f"Loaded new loan data with shape: {df_new.shape}")
except FileNotFoundError:
    print(f"Error: The file '{NEW_LOAN_DATA_PATH}' was not found.")
    exit()

with open(MODEL_ARTIFACTS_PATH, 'rb') as f:
    artifacts = pickle.load(f)
    model = artifacts['model']
    model_columns_woe = artifacts['model_columns']

with open(WOE_TABLES_PATH, 'rb') as f:
    woe_tables = pickle.load(f)
print("✅ Artifacts loaded successfully.")

# 2. Preprocess the New Data
print("\n>>> STEP 2: PREPROCESSING NEW DATA (MUST MATCH TRAINING)...")
# Impute missing values using the SAME constants as the training data
for col, val in IMPUTATION_VALUES.items():
    if col in df_new.columns:
        df_new[col].fillna(val, inplace=True)
        print(f"  - Imputed missing values in '{col}'.")
# Ensure all columns required for the model exist, fill with a neutral value if not
model_features_orig = [c.replace('_woe', '') for c in model_columns_woe]
for col in model_features_orig:
    if col not in df_new.columns:
        df_new[col] = np.nan  # Or an appropriate default
        print(f"Warning: Column '{col}' not in new data, created as empty.")
# In Step 2 of score_new_loans.py
print("  - Creating actual target variable for comparison...")
default_statuses = ['Charged Off', 'Default']
df_new['actual_default_status'] = df_new['loan_status'].isin(default_statuses).astype(int)

# 3. Apply WoE Transformations
print("\n>>> STEP 3: APPLYING SAVED WoE TRANSFORMATIONS...")
df_new_woe = pd.DataFrame()
numerical_vars = df_new.select_dtypes(include=np.number).columns

for feature_woe in model_columns_woe:
    feature_orig = feature_woe.replace('_woe', '')

    if feature_orig not in df_new.columns:
        continue

    woe_map = woe_tables[feature_orig]['WoE'].to_dict()

    if feature_orig in numerical_vars:
        binned_series = optimal_binning_numerical(df_new, feature_orig)
        df_new_woe[feature_woe] = binned_series.map(woe_map)
    else:
        df_new_woe[feature_woe] = df_new[feature_orig].map(woe_map)

    # Fill any values not found in the original WoE map with 0 (neutral WoE)
    df_new_woe[feature_woe] = pd.to_numeric(df_new_woe[feature_woe], errors='coerce').fillna(0)
    print(f"  - Transformed '{feature_orig}' to WoE.")

# Ensure all columns the model expects are present
df_new_woe = df_new_woe.reindex(columns=model_columns_woe, fill_value=0)

# 4. Score the Transformed Data
print("\n>>> STEP 4: SCORING NEW LOANS WITH THE MODEL...")
# Predict the probability of default (class 1)
predicted_probabilities = model.predict_proba(df_new_woe)[:, 1]

# Add the predictions to the original new dataframe
df_new['predicted_default_probability'] = predicted_probabilities
print("✅ Scoring complete.")

# At the end of Step 4
from sklearn import metrics

# Ensure the actuals column exists before trying to evaluate
if 'actual_default_status' in df_new.columns:
    auc_new_data = metrics.roc_auc_score(df_new['actual_default_status'], df_new['predicted_default_probability'])
    gini_new_data = 2 * auc_new_data - 1
    print("\n>>> PERFORMANCE ON NEW TEST DATA <<<")
    print(f"  - AUC Score: {auc_new_data:.4f}")
    print(f"  - Gini Coefficient: {gini_new_data:.4f}")

# 5. Save the Final Scored Data
print(f"\n>>> STEP 5: SAVING SCORED DATA TO '{SCORE_OUTPUT_PATH}'...")
# Select key columns to save for review
output_columns = [
    'id', 'member_id', 'loan_amnt', 'term', 'int_rate', 'grade', 'home_ownership',
    'annual_inc', 'predicted_default_probability'
]
# Filter for columns that actually exist in the test file
output_columns_exist = [col for col in output_columns if col in df_new.columns]

df_new[output_columns_exist].to_csv(SCORE_OUTPUT_PATH, index=False)
print(f"✅ Final results saved. Check '{SCORE_OUTPUT_PATH}'.")

print("\n--- SCRIPT COMPLETE ---")