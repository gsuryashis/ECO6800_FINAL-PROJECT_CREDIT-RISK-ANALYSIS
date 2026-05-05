import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

import os

# --- Configuration ---
# Place lc_loan.csv in the data/raw/ folder, or update this path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
INPUT_FILE = os.path.join(DATA_DIR, 'lc_loan.csv')

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Original dataset shape: {df.shape}")

# Step 1: Create target variable for scorecard modeling
print("\n" + "="*50)
print("STEP 1: CREATE TARGET VARIABLE")
print("="*50)

# Check loan_status values
print("Loan Status Distribution:")
print(df['loan_status'].value_counts())

# Create binary target variable (1 = Bad, 0 = Good)
df['default_status'] = df['loan_status'].isin(['Charged Off', 'Default']).astype(int)
print(f"\nTarget variable 'default_status' created:")
print(f"Good loans (0): {(df['default_status'] == 0).sum():,} ({(df['default_status'] == 0).mean()*100:.2f}%)")
print(f"Bad loans (1): {(df['default_status'] == 1).sum():,} ({(df['default_status'] == 1).mean()*100:.2f}%)")

# Step 2: Drop variables with extremely high missing values (>90%)
print("\n" + "="*50)
print("STEP 2: DROP HIGH MISSING VARIABLES")
print("="*50)

# Variables to drop (>90% missing)
high_missing_vars = [
    'dti_joint', 'annual_inc_joint', 'verification_status_joint',  # Joint application vars
    'il_util', 'mths_since_rcnt_il', 'inq_fi', 'open_rv_24m', 'open_acc_6m',  # Advanced credit vars
    'all_util', 'inq_last_12m', 'total_cu_tl', 'open_il_12m', 'max_bal_bc',
    'open_il_6m', 'open_il_24m', 'open_rv_12m', 'total_bal_il'
]

print(f"Dropping {len(high_missing_vars)} variables with >90% missing values:")
for var in high_missing_vars:
    if var in df.columns:
        missing_pct = (df[var].isnull().sum() / len(df)) * 100
        print(f"  - {var}: {missing_pct:.2f}% missing")

df_cleaned = df.drop(columns=[var for var in high_missing_vars if var in df.columns])
print(f"\nDataset shape after dropping high missing variables: {df_cleaned.shape}")

# Step 3: Drop non-predictive variables for scorecard modeling
print("\n" + "="*50)
print("STEP 3: DROP NON-PREDICTIVE VARIABLES")
print("="*50)

# Variables to drop for scorecard modeling
non_predictive_vars = [
    'id', 'member_id',  # Identifiers
    'url', 'desc',  # Text descriptions (85% missing anyway)
    'pymnt_plan',  # Usually constant
    'policy_code',  # Usually constant
    'application_type',  # Usually constant
    'zip_code',  # Too granular, use addr_state instead
    'title',  # Similar to purpose
    # Note: Keeping payment and credit pull dates - can be useful features
    # Note: Keeping payment amounts - can indicate payment behavior patterns
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',  # These are outcomes, not predictors
    'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt'  # These are post-default outcomes
]

print(f"Dropping {len([v for v in non_predictive_vars if v in df_cleaned.columns])} non-predictive variables:")
for var in non_predictive_vars:
    if var in df_cleaned.columns:
        print(f"  - {var}")

df_cleaned = df_cleaned.drop(columns=[var for var in non_predictive_vars if var in df_cleaned.columns])
print(f"\nDataset shape after dropping non-predictive variables: {df_cleaned.shape}")

# Step 4: Handle moderate missing values
print("\n" + "="*50)
print("STEP 4: HANDLE MODERATE MISSING VALUES")
print("="*50)

# Check remaining missing values
remaining_missing = df_cleaned.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False)
print("Remaining variables with missing values:")
for var, count in remaining_missing.items():
    pct = (count / len(df_cleaned)) * 100
    print(f"  {var}: {count:,} ({pct:.2f}%)")

# Handle specific variables with business logic
print("\nHandling missing values with business logic:")

# 1. emp_length - Create 'Unknown' category
if 'emp_length' in df_cleaned.columns:
    df_cleaned['emp_length'] = df_cleaned['emp_length'].fillna('Unknown')
    print("  - emp_length: Filled with 'Unknown'")

# 2. emp_title - Create 'Not Provided' category
if 'emp_title' in df_cleaned.columns:
    df_cleaned['emp_title'] = df_cleaned['emp_title'].fillna('Not Provided')
    print("  - emp_title: Filled with 'Not Provided'")

# 3. revol_util - Impute with median (small % missing)
if 'revol_util' in df_cleaned.columns:
    median_revol_util = df_cleaned['revol_util'].median()
    df_cleaned['revol_util'] = df_cleaned['revol_util'].fillna(median_revol_util)
    print(f"  - revol_util: Filled with median ({median_revol_util:.2f})")

# 4. Months since variables - Fill with large number (e.g., 999) to indicate "never"
months_since_vars = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']
for var in months_since_vars:
    if var in df_cleaned.columns:
        df_cleaned[var] = df_cleaned[var].fillna(999)
        print(f"  - {var}: Filled with 999 (never occurred)")

# 5. Credit variables - Fill small amounts with median/mode
credit_vars = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'collections_12_mths_ex_med']
for var in credit_vars:
    if var in df_cleaned.columns:
        if df_cleaned[var].dtype in ['int64', 'float64']:
            fill_value = df_cleaned[var].median()
            df_cleaned[var] = df_cleaned[var].fillna(fill_value)
            print(f"  - {var}: Filled with median ({fill_value:.2f})")

# 6. Remaining numeric variables - Fill with median
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_cleaned[col].isnull().sum() > 0:
        fill_value = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(fill_value)
        print(f"  - {col}: Filled with median ({fill_value:.2f})")

# 7. Remaining categorical variables - Fill with most frequent
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'loan_status' and df_cleaned[col].isnull().sum() > 0:
        fill_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
        df_cleaned[col] = df_cleaned[col].fillna(fill_value)
        print(f"  - {col}: Filled with mode ({fill_value})")

# Step 5: Final data quality check
print("\n" + "="*50)
print("STEP 5: FINAL DATA QUALITY CHECK")
print("="*50)

final_missing = df_cleaned.isnull().sum().sum()
print(f"Total missing values remaining: {final_missing}")
print(f"Final dataset shape: {df_cleaned.shape}")

if final_missing == 0:
    print("✅ All missing values handled successfully!")
else:
    print("⚠️ Some missing values remain:")
    remaining = df_cleaned.isnull().sum()
    remaining = remaining[remaining > 0]
    for var, count in remaining.items():
        print(f"  {var}: {count} missing")

# Display final column list
print(f"\nFinal variables for scorecard modeling ({len(df_cleaned.columns)} total):")
for i, col in enumerate(sorted(df_cleaned.columns), 1):
    print(f"{i:2d}. {col}")

# Save cleaned dataset
output_file = 'loan_data_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\n✅ Cleaned dataset saved as: {output_file}")

# Summary statistics
print("\n" + "="*50)
print("PREPROCESSING SUMMARY")
print("="*50)
print(f"Original variables: {df.shape[1]}")
print(f"Final variables: {df_cleaned.shape[1]}")
print(f"Variables dropped: {df.shape[1] - df_cleaned.shape[1]}")
print(f"Records: {df_cleaned.shape[0]:,}")
print(f"Target distribution: {df_cleaned['default_status'].mean()*100:.2f}% bad rate")