import pandas as pd

from utils.poisson_utils.data import detect_current_season


def test_detect_current_season_gap_based():
    dates = pd.to_datetime([
        "2023-03-01",
        "2023-05-01",
        "2023-08-15",
        "2023-09-01",
    ])
    df = pd.DataFrame({"Date": dates, "HomeTeam": ["A"] * 4, "AwayTeam": ["B"] * 4})

    season_df, start = detect_current_season(df, start_month=8, gap_days=30)

    assert start == pd.Timestamp("2023-08-15")
    assert len(season_df) == 2


def test_detect_current_season_fallback_start_month():
    dates = pd.to_datetime([
        "2023-05-10",
        "2023-06-01",
        "2023-06-20",
        "2023-07-15",
        "2023-08-05",
    ])
    df = pd.DataFrame({"Date": dates, "HomeTeam": ["A"] * 5, "AwayTeam": ["B"] * 5})

    season_df, start = detect_current_season(df, start_month=7, gap_days=30)

    assert start == pd.Timestamp("2023-07-01")
    assert len(season_df) == 2
