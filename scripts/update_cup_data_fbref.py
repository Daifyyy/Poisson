import os
import sys

"""
Fetch cup match data from FBref schedule/results pages and save
into CSV files matching the league schema.

Usage:
    python scripts/update_cup_data_fbref.py            # update all cups
    python scripts/update_cup_data_fbref.py FAC DFB    # update selected cups

The mapping `CUP_URLS` defines the FBref schedule/results page for
each competition.  Add or adjust entries as needed.
"""

import pandas as pd
import requests

# Root directory of the repository and the target data folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Mapping of cup codes to FBref schedule/results URLs
CUP_URLS = {
    # FA Cup 2024-2025
    "FAC": "https://fbref.com/en/comps/530/2024-2025/schedule/2024-2025-FA-Cup-Scores-and-Fixtures",
    # DFB-Pokal 2024-2025
    "DFB": "https://fbref.com/en/comps/531/2024-2025/schedule/2024-2025-DFB-Pokal-Scores-and-Fixtures",
    # Coppa Italia 2024-2025
    "CIA": "https://fbref.com/en/comps/532/2024-2025/schedule/2024-2025-Coppa-Italia-Scores-and-Fixtures",
}


def log(msg: str) -> None:
    """Simple logger with flush."""
    print(msg, flush=True)


def fetch_table(url: str) -> pd.DataFrame:
    """Fetch the main schedule/results table from an FBref page."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PoissonBot/1.1)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    for tbl in tables:
        if "Score" in tbl.columns and "Home" in tbl.columns and "Away" in tbl.columns:
            return tbl
    raise ValueError("No valid table found at URL: %s" % url)


def process_table(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Convert a raw FBref table to the league CSV schema."""
    df = df.copy()
    df = df[df["Score"].notna()]
    df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"}, inplace=True)

    # Parse score column into FTHG and FTAG
    score_split = df["Score"].astype(str).str.replace("–", "-", regex=False).str.split("-", expand=True)
    df["FTHG"] = pd.to_numeric(score_split[0], errors="coerce")
    df["FTAG"] = pd.to_numeric(score_split[1], errors="coerce")
    df.dropna(subset=["FTHG", "FTAG"], inplace=True)

    # Full Time Result
    df["FTR"] = df.apply(
        lambda r: "H" if r["FTHG"] > r["FTAG"] else ("A" if r["FTHG"] < r["FTAG"] else "D"),
        axis=1,
    )

    # Date formatting
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    df["Div"] = code
    cols = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    return df[cols]


def update_cup(code: str, url: str) -> None:
    """Fetch and store one competition."""
    log(f"🔄 {code}: {url}")
    try:
        raw = fetch_table(url)
    except Exception as exc:
        log(f"⚠️ Failed to fetch {code}: {exc}")
        return
    df = process_table(raw, code)
    out_path = os.path.join(DATA_DIR, f"{code}_combined_full.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"✅ Saved {len(df)} matches → {out_path}")


def main(codes: list[str] | None = None) -> None:
    target = CUP_URLS if not codes else {c: CUP_URLS[c] for c in codes if c in CUP_URLS}
    for code, url in target.items():
        update_cup(code, url)


if __name__ == "__main__":
    main(sys.argv[1:])
