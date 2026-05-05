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
