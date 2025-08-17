import pandas as pd
import pytest

from utils.poisson_utils import calculate_cross_league_team_index
from utils.poisson_utils.cross_league import WORLD_ELO_MEAN


@pytest.fixture
def sample_european_teams():
    teams = pd.DataFrame(
        {
            "league": ["Premier League", "La Liga", "Eredivisie"],
            "team": ["Liverpool", "Real Madrid", "PSV"],
            "matches": [10, 10, 10],
            "goals_for": [20, 20, 20],
            "goals_against": [10, 10, 10],
            "xg_for": [18, 18, 18],
            "xg_against": [9, 9, 9],
        }
    )
    ratings = pd.DataFrame(
        {
            "league": ["Premier League", "La Liga", "Eredivisie"],
            "elo": [1700, 1680, 1500],
        }
    )
    matches = pd.DataFrame(
        {
            "Date": pd.date_range("2021-08-01", periods=6, freq="D"),
            "league": [
                "Premier League",
                "Premier League",
                "La Liga",
                "La Liga",
                "Eredivisie",
                "Eredivisie",
            ],
            "HomeTeam": [
                "Liverpool",
                "Everton",
                "Real Madrid",
                "Sevilla",
                "PSV",
                "Ajax",
            ],
            "AwayTeam": [
                "Everton",
                "Liverpool",
                "Sevilla",
                "Real Madrid",
                "Ajax",
                "PSV",
            ],
            "FTHG": [1, 1, 1, 1, 1, 1],
            "FTAG": [1, 1, 1, 1, 1, 1],
        }
    )
    return teams, ratings, matches


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

    # league penalty coefficient should be exposed
    assert "league_penalty_coef" in result.columns
    a1_coef = result.loc[result["team"] == "A1", "league_penalty_coef"].item()
    assert a1_coef == pytest.approx(1600 / WORLD_ELO_MEAN, rel=1e-3)


def test_cross_league_team_index_respects_league_strength(sample_european_teams):
    teams, ratings, matches = sample_european_teams
    result = calculate_cross_league_team_index(teams, ratings, matches)
    ordered = list(result.sort_values("team_index", ascending=False)["team"])
    assert ordered == ["Liverpool", "Real Madrid", "PSV"]


def test_cross_league_team_index_falls_back_to_goals_when_xg_missing():
    teams = pd.DataFrame(
        {
            "league": ["A", "A"],
            "team": ["A1", "A2"],
            "matches": [2, 2],
            "goals_for": [3, 0],
            "goals_against": [0, 3],
        }
    )
    ratings = pd.DataFrame({"league": ["A"], "elo": [1500]})
    matches = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=2, freq="D"),
            "league": ["A", "A"],
            "HomeTeam": ["A1", "A2"],
            "AwayTeam": ["A2", "A1"],
            "FTHG": [2, 0],
            "FTAG": [0, 1],
        }
    )

    with pytest.warns(UserWarning):
        result = calculate_cross_league_team_index(teams, ratings, matches)

    a1 = result.loc[result["team"] == "A1"].iloc[0]
    expected_vs_world = a1["goals_for"] - a1["goals_against"]
    assert a1["xg_vs_world"] == pytest.approx(expected_vs_world, rel=1e-3)
    assert a1["xg_diff_norm"] != 0


def test_cross_league_team_index_requires_xg_or_goals():
    teams = pd.DataFrame({"league": ["A"], "team": ["A1"], "matches": [1]})
    ratings = pd.DataFrame({"league": ["A"], "elo": [1500]})
    matches = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=1),
            "league": ["A"],
            "HomeTeam": ["A1"],
            "AwayTeam": ["A1"],
            "FTHG": [0],
            "FTAG": [0],
        }
    )

    with pytest.raises(ValueError):
        calculate_cross_league_team_index(teams, ratings, matches)


def test_cross_league_team_index_accepts_precomputed_penalty():
    teams = pd.DataFrame(
        {
            "league": ["A", "A"],
            "team": ["A1", "A2"],
            "matches": [1, 1],
            "goals_for": [1, 0],
            "goals_against": [0, 1],
            "xg_for": [1.2, 0.8],
            "xg_against": [0.8, 1.2],
        }
    )
    ratings = pd.DataFrame({"league": ["A"], "elo": [1500], "penalty_coef": [0.8]})
    matches = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=1),
            "league": ["A"],
            "HomeTeam": ["A1"],
            "AwayTeam": ["A2"],
            "FTHG": [1],
            "FTAG": [0],
        }
    )
    result = calculate_cross_league_team_index(teams, ratings, matches)
    assert result["league_penalty_coef"].eq(0.8).all()
    a1 = result.loc[result["team"] == "A1"].iloc[0]
    expected_vs_world = (a1["xg_for"] - a1["xg_against"]) * 0.8
    assert a1["xg_vs_world"] == pytest.approx(expected_vs_world, rel=1e-3)
