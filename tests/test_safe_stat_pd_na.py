import pandas as pd

from utils.poisson_utils.team_analysis import (
    expected_goals_combined_homeaway_allmatches,
    calculate_expected_and_actual_points,
)


def test_expected_goals_handles_pandas_na():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"]),
            "HomeTeam": ["Existing"],
            "AwayTeam": ["Other"],
            "FTHG": pd.Series([1], dtype="Int64"),
            "FTAG": pd.Series([1], dtype="Int64"),
        }
    )

    elo = {"Missing": 1500, "Existing": 1500, "Other": 1500}

    home, away = expected_goals_combined_homeaway_allmatches(
        df, "Missing", "Existing", elo
    )

    assert isinstance(home, float)
    assert isinstance(away, float)


def test_expected_points_handles_pandas_na():
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "HomeTeam": ["Existing"],
            "AwayTeam": ["Other"],
            "FTHG": [1],
            "FTAG": [1],
            "HS": [pd.NA],
            "HST": [pd.NA],
            "AS": [pd.NA],
            "AST": [pd.NA],
        }
    )

    res = calculate_expected_and_actual_points(df)

    assert res["Existing"]["points"] == 1
    assert res["Other"]["points"] == 1
    assert isinstance(res["Existing"]["expected_points"], float)
    assert isinstance(res["Other"]["expected_points"], float)
