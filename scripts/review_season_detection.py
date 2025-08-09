"""Utility to inspect detected seasons across league datasets."""
from __future__ import annotations

from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from utils.poisson_utils import load_data, detect_current_season

# Mapping of league names to data files, mirroring the main application.
LEAGUE_FILES = {
    "E0 (Premier League)": "data/E0_combined_full_updated.csv",
    "E1 (Championship)": "data/E1_combined_full_updated.csv",
    "SP1 (La Liga)": "data/SP1_combined_full_updated.csv",
    "B1 (Jupiler League)": "data/B1_combined_full_updated.csv",
    "D1 (Bundesliga)": "data/D1_combined_full_updated.csv",
    "D2 (2. Bundesliga)": "data/D2_combined_full_updated.csv",
    "I1 (Seria A)": "data/I1_combined_full_updated.csv",
    "F1 (Ligue 1)": "data/F1_combined_full_updated.csv",
    "N1 (Eredivisie)": "data/N1_combined_full_updated.csv",
    "P1 (Primeira Liga)": "data/P1_combined_full_updated.csv",
    "T1 (Super League)": "data/T1_combined_full_updated.csv",
}

# Months when each league typically starts a new season. Leagues not listed
# default to August.
LEAGUE_START_MONTH = {
    "B1": 7,  # Jupiler League begins in July
    "D2": 7,  # 2. Bundesliga kicks off in July
}


def main() -> None:
    for name, rel_path in LEAGUE_FILES.items():
        file_path = root / rel_path
        df = load_data(file_path)
        league_code = Path(rel_path).name.split("_")[0]
        start_month = LEAGUE_START_MONTH.get(league_code, 8)
        season_df, season_start = detect_current_season(df, start_month=start_month)
        print(f"{name}: {len(season_df)} matches since {season_start.date()}")


if __name__ == "__main__":
    main()
