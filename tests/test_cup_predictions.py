import pandas as pd
import pytest

from utils.poisson_utils import predict_cup_match
from utils.poisson_utils import cup_predictions


@pytest.fixture
def cross_league_df():
    return pd.DataFrame(
        {
            "team": ["CL Team", "EL Team", "ECL Team"],
            "league": ["CL", "EL", "ECL"],
            "team_index": [1.2, 1.0, 0.8],
        }
    )


def test_probabilities_sum_and_rating_effect(cross_league_df):
    preds = predict_cup_match("CL Team", "ECL Team", cross_league_df)
    total = preds["home_win_pct"] + preds["draw_pct"] + preds["away_win_pct"]
    assert total == pytest.approx(100, abs=0.5)
    assert preds["home_win_pct"] > preds["away_win_pct"]

    reverse = predict_cup_match("ECL Team", "CL Team", cross_league_df)
    total_rev = reverse["home_win_pct"] + reverse["draw_pct"] + reverse["away_win_pct"]
    assert total_rev == pytest.approx(100, abs=0.5)
    assert reverse["away_win_pct"] > reverse["home_win_pct"]


def test_neutral_venue_identical_ratings(monkeypatch):
    df = pd.DataFrame(
        {
            "team": ["Team A", "Team B"],
            "league": ["CL", "EL"],
            "team_index": [1.0, 1.0],
        }
    )

    monkeypatch.setattr(cup_predictions, "HOME_ADVANTAGE", 0)

    preds = predict_cup_match("Team A", "Team B", df)
    total = preds["home_win_pct"] + preds["draw_pct"] + preds["away_win_pct"]
    assert total == pytest.approx(100, abs=0.5)
    assert preds["home_win_pct"] == pytest.approx(preds["away_win_pct"], abs=1e-2)
