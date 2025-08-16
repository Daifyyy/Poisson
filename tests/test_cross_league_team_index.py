import pandas as pd
import pytest

from utils.poisson_utils import calculate_cross_league_team_index


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

    # team_index should rank A1 highest, B1 second, A2 lowest
    ordered = list(result.sort_values("team_index", ascending=False)["team"])
    assert ordered == ["A1", "B1", "A2"]

    # check expected value for A1 approximately
    a1_index = result.loc[result["team"] == "A1", "team_index"].item()
    assert a1_index == pytest.approx(21.94, rel=1e-2)
