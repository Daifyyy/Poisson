"""
This version of the league updater handles multiple seasons per league to ensure accurate historical data,
especially for reconstructing files with invalid 'Date' fields.
"""
import os
import sys
import time
import io
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ⚠️ Season ranges to include
SEASON_CODES = ["2526", "2425", "2324"]  # Adjust as needed
LEAGUE_CODES = ["E0", "E1", "D1", "D2", "SP1", "I1", "F1", "N1", "P1", "B1"]

EXPECTED_COLS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "Avg>2.5", "Avg<2.5"
]

def log(msg): print(msg, flush=True)

def flip_scheme(url):
    if url.startswith("https://"): return "http://" + url[len("https://"):]
    if url.startswith("http://"): return "https://" + url[len("http://"):]
    return url

def robust_get(url, tries=4, timeout=20):
    last_exc = None; flipped_once = False
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (compatible; PoissonBot/1.1)"
            })
            if r.status_code == 200 and r.content:
                return r
            if attempt >= 2 and not flipped_once:
                url = flip_scheme(url); flipped_once = True
            time.sleep(1.0 * attempt)
        except Exception as e:
            last_exc = e; time.sleep(1.0 * attempt)
    return None

def normalize_columns(cols):
    return pd.Index(cols).astype(str).str.strip().str.replace("﻿", "", regex=False).tolist()

def _looks_like_csv(txt):
    first_line = next((ln for ln in txt.splitlines() if ln.strip()), "")
    return "HomeTeam" in first_line or first_line.count(",") >= 5

def read_csv_safely(content, expected_div, content_type=""):
    head = content[:1024].decode("utf-8", errors="ignore").lower()
    txt = content.decode("utf-8-sig", errors="ignore")
    if not _looks_like_csv(txt): return pd.DataFrame()
    buf = io.StringIO(txt)
    df = pd.read_csv(buf, sep=",", engine="python")
    df.columns = normalize_columns(df.columns)
    if "Div" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame()
    df = df[df["Div"].astype(str).str.strip() == expected_div]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df[~df["Date"].isna()].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df

def generate_key(df):
    return df["Date"].astype(str).str[:10] + "|" + df["HomeTeam"].str.lower().str.strip() + "|" + df["AwayTeam"].str.lower().str.strip()

def update_league_all_seasons(code):
    combined = []
    for season_code in SEASON_CODES:
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
        log(f"🔄 {code} {season_code}: {url}")
        resp = robust_get(url)
        if not resp:
            log(f"⚠️ Failed to fetch {url}"); continue
        df = read_csv_safely(resp.content, expected_div=code, content_type=resp.headers.get("Content-Type", ""))
        if df.empty:
            log(f"⚠️ No valid data in {url}"); continue
        df["SeasonCode"] = season_code
        combined.append(df)

    if not combined:
        log(f"❌ No data to combine for {code}")
        return

    df_all = pd.concat(combined, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date")
    # Retain only expected columns if present
    expected_cols = [
        "Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HF", "AF",
        "HC", "AC", "HY", "AY", "HR", "AR"
    ]
    df_all = df_all[[c for c in expected_cols if c in df_all.columns]]
    out_path = os.path.join(DATA_DIR, f"{code}_combined_full_updated.csv")
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"✅ Saved {len(df_all)} matches → {out_path}")

def update_all():
    for code in LEAGUE_CODES:
        update_league_all_seasons(code)

if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        for code in args:
            if code in LEAGUE_CODES:
                update_league_all_seasons(code)
            else:
                log(f"Unknown league code: {code}")
    else:
        update_all()
