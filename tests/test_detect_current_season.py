import sys
import pathlib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils.poisson_utils.data import detect_current_season


def test_detect_current_season_identifies_latest_break():
    df = pd.DataFrame({
        "Date": [
            "30/05/2024",
            "15/08/2024",
            "22/08/2024",
            "29/08/2024",
        ],
        "HomeTeam": ["A", "B", "C", "D"],
        "AwayTeam": ["E", "F", "G", "H"],
    })

    season_df, season_start = detect_current_season(df, gap_days=30)

    assert season_start == pd.Timestamp("2024-08-15")
    assert list(season_df["Date"]) == [
        pd.Timestamp("2024-08-15"),
        pd.Timestamp("2024-08-22"),
        pd.Timestamp("2024-08-29"),
    ]
