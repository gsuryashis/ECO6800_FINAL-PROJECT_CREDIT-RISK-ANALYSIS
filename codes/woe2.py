import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import pickle # Add this import at the top of the file

# --- Configuration ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)

# --- File Paths ---
INPUT_CSV = 'loan_data_cleaned.csv'
OUTPUT_WOE_CSV = 'loan_data_woe_transformed.csv'
OUTPUT_IV_CSV = 'variable_selection_results.csv'

# --- Variable Lists for Removal ---
# These variables are removed before any analysis is done.
VARS_TO_REMOVE = {
    # 1. Multicollinear variables identified in EDA
    'multicollinear': [
        'funded_amnt',
        'funded_amnt_inv',
        'out_prncp_inv',
        'total_pymnt_inv'
    ],
    # 2. Data leakage (info not available at application time) or suspicious IV
    'data_leakage': [
        'last_pymnt_d',  # Highly predictive of default, but occurs post-application
        'out_prncp',  # Directly indicates if a loan is active/paid off
        'issue_d',  # Can capture underwriting changes, not borrower risk
        'next_pymnt_d'  # Also post-application information
    ],
    # 3. High cardinality / problematic variables
    'problematic': [
        'emp_title',  # Too many unique values to be useful
        'loan_status'  # The source of the target, must be removed
    ]
}

TARGET_VARIABLE = 'default_status'
MAX_CATEGORICAL_UNIQUE_VALUES = 100  # Skip categorical variables with more than this many unique values


# --- Function Definitions ---

def calculate_woe_iv(df, feature, target):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) for a feature.
    """
    # Create a contingency table (crosstab)
    crosstab = pd.crosstab(df[feature], df[target], margins=False)

    # Add a small number to prevent division by zero
    crosstab = crosstab + 0.0001

    # Calculate totals
    total_good = crosstab[0].sum()
    total_bad = crosstab[1].sum()

    # Calculate rates
    crosstab['good_rate'] = crosstab[0] / total_good
    crosstab['bad_rate'] = crosstab[1] / total_bad

    # Calculate WoE
    crosstab['WoE'] = np.log(crosstab['good_rate'] / crosstab['bad_rate'])

    # Calculate IV
    crosstab['IV'] = (crosstab['good_rate'] - crosstab['bad_rate']) * crosstab['WoE']

    return crosstab, crosstab['IV'].sum()


def optimal_binning_numerical(df, feature, target, max_bins=10):
    """
    Creates optimal bins for numerical variables using quantiles and returns a binned series.
    """
    # Handle cases with few unique values
    if df[feature].nunique() <= max_bins:
        return df[feature]

    # Generate quantile-based bin edges
    try:
        bins = pd.qcut(df[feature], q=max_bins, retbins=True, duplicates='drop')[1]
    except ValueError:
        # If quantile binning fails, use equal-width bins
        bins = np.linspace(df[feature].min(), df[feature].max(), max_bins + 1)

    # Ensure bin edges are unique
    bins = np.unique(bins)

    # Set infinite bounds to capture all values
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Create binned variable
    return pd.cut(df[feature], bins=bins)


# --- Main Script ---

# 1. Load and Prepare Data
print(">>> STEP 1: LOADING AND PREPARING DATA...")
df = pd.read_csv(INPUT_CSV)
print(f"Original dataset shape: {df.shape}")

# Remove all specified variables
all_vars_to_remove = [item for sublist in VARS_TO_REMOVE.values() for item in sublist]
df_processed = df.drop(columns=all_vars_to_remove, errors='ignore')

print(f"Dropped {len(all_vars_to_remove)} variables.")
print(f"Dataset shape after removal: {df_processed.shape}\n")

# 2. Calculate WoE and IV
print(">>> STEP 2: CALCULATING WEIGHT OF EVIDENCE (WoE) & INFORMATION VALUE (IV)...")
woe_iv_results = {}
woe_tables = {}

numerical_vars = df_processed.select_dtypes(include=np.number).columns.tolist()
numerical_vars.remove(TARGET_VARIABLE)
categorical_vars = df_processed.select_dtypes(include=['object']).columns.tolist()

# Process numerical variables
for var in numerical_vars:
    binned_series = optimal_binning_numerical(df_processed, var, TARGET_VARIABLE)
    temp_df = pd.DataFrame({var: binned_series, TARGET_VARIABLE: df_processed[TARGET_VARIABLE]})
    woe_table, iv_score = calculate_woe_iv(temp_df, var, TARGET_VARIABLE)
    woe_iv_results[var] = iv_score
    woe_tables[var] = woe_table

# Process categorical variables
for var in categorical_vars:
    if df_processed[var].nunique() > MAX_CATEGORICAL_UNIQUE_VALUES:
        print(f"  - Skipping '{var}': Too many unique values ({df_processed[var].nunique()})")
        continue
    woe_table, iv_score = calculate_woe_iv(df_processed, var, TARGET_VARIABLE)
    woe_iv_results[var] = iv_score
    woe_tables[var] = woe_table

print("WoE and IV calculation complete.\n")

# 3. Variable Selection based on IV
print(">>> STEP 3: SELECTING VARIABLES BASED ON IV SCORES...")
iv_df = pd.DataFrame.from_dict(woe_iv_results, orient='index', columns=['IV_Score'])
iv_df = iv_df.sort_values(by='IV_Score', ascending=False)


# Define predictive power categories
def get_power(iv):
    if iv < 0.02:
        return "Not useful"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    elif iv < 0.5:
        return "Strong"
    else:
        return "Suspicious"


iv_df['Predictive_Power'] = iv_df['IV_Score'].apply(get_power)

# Select variables with at least weak predictive power for the model
selected_variables = iv_df[(iv_df['IV_Score'] >= 0.02) & (iv_df['IV_Score'] < 0.5)].index.tolist()

print("Information Value Summary:")
print(iv_df)
print(f"\nSelected {len(selected_variables)} variables for modeling (IV between 0.02 and 0.5).\n")

# 4. Create WoE Transformed Dataset
print(">>> STEP 4: CREATING THE WoE TRANSFORMED DATASET...")
df_woe = df_processed[[TARGET_VARIABLE]].copy()

for var in selected_variables:
    woe_map = woe_tables[var]['WoE'].to_dict()

    if var in numerical_vars:
        binned_series = optimal_binning_numerical(df_processed, var, TARGET_VARIABLE)
        df_woe[f'{var}_woe'] = binned_series.map(woe_map)
    else:
        df_woe[f'{var}_woe'] = df_processed[var].map(woe_map)

    # **FIX:** Convert column to numeric before filling NaNs
    df_woe[f'{var}_woe'] = pd.to_numeric(df_woe[f'{var}_woe'], errors='coerce').fillna(0)
    print(f"  - Transformed '{var}' to WoE.")

print(f"\nWoE transformed dataset created with shape: {df_woe.shape}\n")

# 5. Save Outputs and Visualize
print(">>> STEP 5: SAVING RESULTS AND VISUALIZING...")

# Save datasets
df_woe.to_csv(OUTPUT_WOE_CSV, index=False)
iv_df.to_csv(OUTPUT_IV_CSV)
print(f"✅ WoE transformed data saved to '{OUTPUT_WOE_CSV}'")
print(f"✅ IV results saved to '{OUTPUT_IV_CSV}'")

# Plot IV scores of selected variables
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

plot_data = iv_df.loc[selected_variables]
sns.barplot(x='IV_Score', y=plot_data.index, data=plot_data, palette='viridis', ax=ax)

ax.set_title('Information Value of Selected Variables', fontsize=16)
ax.set_xlabel('Information Value (IV)', fontsize=12)
ax.set_ylabel('Variable', fontsize=12)
ax.axvline(x=0.1, color='grey', linestyle='--', label='Weak Threshold (0.1)')
ax.axvline(x=0.3, color='red', linestyle='--', label='Strong Threshold (0.3)')
ax.legend()

plt.tight_layout()
plt.show()

print("\n--- SCRIPT COMPLETE ---")
print("Next step is to use the 'loan_data_woe_transformed.csv' file to build your logistic regression model.")
# --- At the end of woe2.py ---

# Add this line before the script finishes
with open('woe_tables.pkl', 'wb') as f:
    pickle.dump(woe_tables, f)

print("✅ WoE tables saved to 'woe_tables.pkl'")