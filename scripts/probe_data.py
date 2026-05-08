# scripts/probe_data.py
import pandas as pd
import os

DATA_PATH = os.path.join("data", "raw", "lc_loan.csv")
FALLBACK_DATA_PATH = os.path.join("data", "fallback", "lc_loan.csv")


def resolve_data_path():
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    if os.path.exists(FALLBACK_DATA_PATH):
        return FALLBACK_DATA_PATH
    raise FileNotFoundError(
        f"❌ Data missing. Expected either {DATA_PATH} or {FALLBACK_DATA_PATH}"
    )

def probe_data():
    data_path = resolve_data_path()
    df = pd.read_csv(data_path, nrows=5)
    required_cols = ["loan_status", "loan_amnt", "dti", "grade"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing expected columns: {missing}")
    print(f"✅ Data loaded from {data_path}. Columns present: {df[required_cols].columns.tolist()}")
    return True

if __name__ == "__main__":
    probe_data()
