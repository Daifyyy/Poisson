import pandas as pd
import pytest

from utils.poisson_utils import (
    calculate_cross_league_team_index,
    CROSS_LEAGUE_COLUMN_LEGEND,
)


def test_cross_league_team_index_basic():
    teams = pd.DataFrame(
        {
            "league": ["A", "A", "B"],
            "team": ["A1", "A2", "B1"],
            "matches": [2, 2, 2],
            "goals_for": [4, 2, 3],
            "goals_against": [1, 2, 2],
            "xg_for": [3.0, 1.0, 2.0],
            "xg_against": [1.0, 2.0, 1.0],
        }
    )
    ratings = pd.DataFrame({"league": ["A", "B"], "elo": [1600, 1400]})

    result = calculate_cross_league_team_index(teams, ratings)

    # cross_league_index should rank A1 highest, B1 second, A2 lowest
    ordered = list(result.sort_values("cross_league_index", ascending=False)["team"])
    assert ordered == ["A1", "B1", "A2"]

    # check expected value for A1 approximately
    a1_index = result.loc[result["team"] == "A1", "cross_league_index"].item()
    assert a1_index == pytest.approx(21.94, rel=1e-2)


def test_columns_have_legend():
    teams = pd.DataFrame(
        {
            "league": ["A"],
            "team": ["A1"],
            "matches": [2],
            "goals_for": [4],
            "goals_against": [1],
            "xg_for": [3.0],
            "xg_against": [1.0],
        }
    )
    ratings = pd.DataFrame({"league": ["A"], "elo": [1600]})

    result = calculate_cross_league_team_index(teams, ratings)
    # ensure each returned metric is documented in legend
    for col in result.columns:
        if col not in {"league", "team", "matches", "elo"}:  # original fields
            assert col in CROSS_LEAGUE_COLUMN_LEGEND
