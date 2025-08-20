#!/usr/bin/env python3
"""
Stáhne zápasy Ligy mistrů (CL), Evropské ligy (EL) a Konferenční ligy (ECL)
z Football-Data.org pro sezóny 2024 a 2025.

Uloží CSV soubory do složky data/ ve formátu:
- CL_combined_full.csv
- EL_combined_full.csv
- ECL_combined_full.csv
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://api.football-data.org/v4"
CUPS = {
    "CL": "CL_combined_full.csv",
    "EL": "EL_combined_full.csv",
    "ECL": "ECL_combined_full.csv",
}


def fetch_matches(code: str, season: int, token: str) -> pd.DataFrame:
    url = f"{BASE_URL}/competitions/{code}/matches?season={season}"
    headers = {"X-Auth-Token": token}
    print(f"🔄 Stahuji {code} sezónu {season}...")

    resp = requests.get(url, headers=headers, timeout=30)

    if resp.status_code == 404:
        print(f"⚠️  Data pro {code} sezóna {season} nejsou dostupná (404 Not Found)")
        return pd.DataFrame()

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"❌ Chyba při volání API: {e}")
        return pd.DataFrame()

    rows = []
    for match in resp.json().get("matches", []):
        rows.append(
            {
                "Date": match["utcDate"][:10],
                "HomeTeam": match["homeTeam"]["name"],
                "AwayTeam": match["awayTeam"]["name"],
                "FTHG": match["score"]["fullTime"]["home"],
                "FTAG": match["score"]["fullTime"]["away"],
                "Season": season,
                "Competition": code,
            }
        )
    return pd.DataFrame(rows)


def seasons_to_fetch() -> list[int]:
    """Vrací sezóny: aktuální a předchozí"""
    return [2024, 2025]


def main() -> None:
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("❌ FOOTBALL_DATA_TOKEN není nastaven.")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for code, filename in CUPS.items():
        all_frames = []

        for season in seasons_to_fetch():
            df = fetch_matches(code, season, token)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            print(f"⚠️  Žádná data nenalezena pro soutěž {code}")
            continue

        full_df = pd.concat(all_frames, ignore_index=True)
        full_df.sort_values(["Season", "Date"], inplace=True)
        path = data_dir / filename
        full_df.to_csv(path, index=False)

        print(f"✅ Uloženo {len(full_df)} zápasů do {path}")


if __name__ == "__main__":
    main()
