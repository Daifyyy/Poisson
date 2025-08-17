import pandas as pd
import pytest

from utils.poisson_utils import calculate_cross_league_team_index


def test_cross_league_team_index_basic():
    teams = pd.DataFrame(
        {
            "league": ["A", "A", "B", "B"],
            "team": ["A1", "A2", "B1", "B2"],
            "matches": [2, 2, 2, 2],
            "goals_for": [3, 0, 2, 0],
            "goals_against": [0, 3, 0, 2],
            "xg_for": [2.5, 0.7, 1.8, 0.6],
            "xg_against": [0.5, 1.3, 0.6, 1.4],
        }
    )
    ratings = pd.DataFrame({"league": ["A", "B"], "elo": [1600, 1400]})
    matches = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=4, freq="D"),
            "league": ["A", "A", "B", "B"],
            "HomeTeam": ["A1", "A2", "B1", "B2"],
            "AwayTeam": ["A2", "A1", "B2", "B1"],
            "FTHG": [2, 0, 1, 0],
            "FTAG": [0, 1, 0, 1],
        }
    )

    result = calculate_cross_league_team_index(teams, ratings, matches)

    # team_index should rank A1 highest, B1 second, A2 third, B2 last
    ordered = list(result.sort_values("team_index", ascending=False)["team"])
    assert ordered == ["A1", "B1", "A2", "B2"]

    # check expected value for A1 approximately
    a1_index = result.loc[result["team"] == "A1", "team_index"].item()
    assert a1_index == pytest.approx(0.810697, rel=1e-3)

    # small sample sizes should be blended toward league averages
    a1_goals_for = result.loc[result["team"] == "A1", "goals_for"].item()
    assert a1_goals_for == pytest.approx(0.9, rel=1e-3)

    # offensive and defensive ratings based on goal/xG differentials
    a1_off = result.loc[result["team"] == "A1", "off_rating"].item()
    a1_def = result.loc[result["team"] == "A1", "def_rating"].item()
    assert a1_off == pytest.approx(0.12, rel=1e-3)
    assert a1_def == pytest.approx(0.095, rel=1e-3)
