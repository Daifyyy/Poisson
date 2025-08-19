#!/usr/bin/env python3
"""Fetch UEFA cup fixtures from Football-Data.org.

This script downloads match data for the Champions League, Europa League
and Europa Conference League for the current season and the two preceding
seasons. The data is written to CSV files in ``data/`` using the naming
scheme expected by the app (e.g. ``CL_combined_full.csv``).

Example
-------
    FOOTBALL_DATA_TOKEN=... python scripts/update_uefa_cups.py

The API token must be supplied via the ``FOOTBALL_DATA_TOKEN`` environment
variable. A free token can be obtained from https://www.football-data.org/.
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
    """Return matches for a competition and season.

    Parameters
    ----------
    code : str
        Competition code used by the API (``CL``, ``EL`` or ``ECL``).
    season : int
        Starting year of the season (e.g. ``2023`` for 2023/24).
    token : str
        Football-Data API token.
    """

    url = f"{BASE_URL}/competitions/{code}/matches?season={season}"
    headers = {"X-Auth-Token": token}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
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
            }
        )
    return pd.DataFrame(rows)


def seasons_to_fetch(reference: int | None = None) -> Iterable[int]:
    """Return the current season and the two preceding seasons."""

    year = reference or dt.date.today().year
    return [year - 2, year - 1, year]


def main() -> None:
    token = os.environ["FOOTBALL_DATA_TOKEN"]
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for code, filename in CUPS.items():
        frames = [fetch_matches(code, s, token) for s in seasons_to_fetch()]
        df = pd.concat(frames, ignore_index=True)
        df.sort_values(["Season", "Date"], inplace=True)
        df.to_csv(data_dir / filename, index=False)
        print(f"Saved {len(df)} matches → {data_dir / filename}")


if __name__ == "__main__":
    main()
