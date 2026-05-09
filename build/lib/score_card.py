import os
import pandas as pd
import numpy as np
import pickle

# --- Configuration ---
_BASE = os.path.join(os.path.dirname(__file__), '..')
MODEL_ARTIFACTS_PATH = os.path.join(_BASE, 'outputs', 'artifacts', 'scorecard_artifacts.pkl')
WOE_TABLES_PATH = os.path.join(_BASE, 'outputs', 'artifacts', 'woe_tables.pkl')
SCORECARD_PATH = os.path.join(_BASE, 'outputs', 'tables', 'final_scorecard.csv')

# Scorecard Scaling Parameters
TARGET_SCORE = 600
TARGET_ODDS = 50
PDO = 20

# --- Scaling Calculations ---
FACTOR = PDO / np.log(2)
OFFSET = TARGET_SCORE - (FACTOR * np.log(TARGET_ODDS))

# --- Main Script ---

# 1. Load Model and WoE Tables
print(">>> STEP 1: LOADING MODEL AND WoE TABLES...")
with open(MODEL_ARTIFACTS_PATH, 'rb') as f:
    artifacts = pickle.load(f)
    model = artifacts['model']
    model_columns = artifacts['model_columns']

with open(WOE_TABLES_PATH, 'rb') as f:
    woe_tables = pickle.load(f)

print("✅ Model and WoE tables loaded successfully.")

# 2. Extract Coefficients
intercept = model.intercept_[0]
coefficients = model.coef_[0]
coeff_df = pd.DataFrame({'Feature': model_columns, 'Coefficient': coefficients})
num_features = len(model_columns)

# 3. Calculate Scorecard Points
print("\n>>> STEP 2: GENERATING SCORECARD POINTS...")
scorecard_list = []
base_score = np.round(OFFSET - FACTOR * (intercept / num_features))

for feature_woe in model_columns:
    # Get original feature name and its coefficient
    feature = feature_woe.replace('_woe', '')
    coeff = coeff_df[coeff_df['Feature'] == feature_woe]['Coefficient'].iloc[0]

    # Get the corresponding WoE table
    woe_table = woe_tables[feature]

    # Calculate points for each bin/category
    woe_table['Points'] = np.round(-FACTOR * (woe_table['WoE'] * coeff))

    # Add base score to each row
    woe_table['Points'] += base_score

    # Format for final table
    scorecard_part = woe_table[['Points']].reset_index()
    scorecard_part.columns = ['Category', 'Points']
    scorecard_part['Feature'] = feature
    scorecard_list.append(scorecard_part[['Feature', 'Category', 'Points']])

# Combine all parts into a single scorecard dataframe
final_scorecard = pd.concat(scorecard_list, ignore_index=True)

print("✅ Scorecard generated successfully.")

# 4. Save and Display the Scorecard
print("\n>>> FINAL SCORECARD <<<")
print(final_scorecard.to_string())

final_scorecard.to_csv(SCORECARD_PATH, index=False)
print(f"\n✅ Final scorecard saved to '{SCORECARD_PATH}'")