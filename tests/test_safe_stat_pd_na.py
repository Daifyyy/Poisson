import pandas as pd

from utils.poisson_utils.team_analysis import (
    expected_goals_combined_homeaway_allmatches,
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
