import pandas as pd
from utils.poisson_utils import load_cup_team_stats


def test_load_cup_team_stats_basic():
    cup_matches = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-01")],
            "HomeTeam": ["A1"],
            "AwayTeam": ["B1"],
            "FTHG": [2],
            "FTAG": [1],
            "HomeLeague": ["A"],
            "AwayLeague": ["B"],
        }
    )
    team_map = {"A1": "A", "B1": "B"}
    stats = load_cup_team_stats(team_map, matches_df=cup_matches)
    assert set(stats.columns) == {
        "league",
        "team",
        "matches",
        "goals_for",
        "goals_against",
        "xg_for",
        "xg_against",
        "shots_for",
        "shots_against",
    }
    assert len(stats) == 2
    a1 = stats[stats["team"] == "A1"].iloc[0]
    b1 = stats[stats["team"] == "B1"].iloc[0]
    assert a1["league"] == "A" and b1["league"] == "B"
    assert a1["goals_for"] == 2 and b1["goals_for"] == 1
    assert a1["goals_against"] == 1 and b1["goals_against"] == 2
    assert a1["xg_for"] == 2 and a1["xg_against"] == 1
    assert b1["xg_for"] == 1 and b1["xg_against"] == 2
    assert (stats["shots_for"] == 0).all() and (stats["shots_against"] == 0).all()
