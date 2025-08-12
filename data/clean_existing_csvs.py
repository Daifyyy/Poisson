# clean_existing_csvs.py
import os
import pandas as pd
from datetime import datetime
from shutil import copy2

# Konfigurace
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BACKUP_DIR = os.path.join(DATA_DIR, "backup_before_clean")
EXPECTED_COLS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "Avg>2.5", "Avg<2.5"
]

os.makedirs(BACKUP_DIR, exist_ok=True)

def log(msg: str):
    print(msg, flush=True)

def validate_and_filter_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    except Exception as e:
        log(f"[ERROR] Failed to read {filepath}: {e}")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    df = df[[col for col in EXPECTED_COLS if col in df.columns]]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df[~df["Date"].isna()]
        df = df[df["Date"] >= pd.Timestamp(datetime.now().year - 3, 1, 1)]
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    else:
        log(f"[WARNING] Missing 'Date' column in {filepath}, skipping.")
        return pd.DataFrame()

    df = df.dropna(how="all")
    return df.reset_index(drop=True)

def process_all_csvs():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    for file in csv_files:
        if not file.endswith("_combined_full_updated.csv"):
            continue

        path = os.path.join(DATA_DIR, file)
        backup_path = os.path.join(BACKUP_DIR, file)
        log(f"[INFO] Processing {file}...")

        df = validate_and_filter_csv(path)
        if df.empty:
            log(f"[SKIP] No valid data found in {file}")
            continue

        # Záloha a přepsání
        if os.path.exists(path):
            copy2(path, backup_path)
            log(f"[INFO] Backup created: {backup_path}")

        df.to_csv(path, index=False)
        log(f"[DONE] Cleaned and saved: {path}")

if __name__ == "__main__":
    process_all_csvs()
