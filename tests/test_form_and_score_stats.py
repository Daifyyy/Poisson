import sys
import pathlib
import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils.poisson_utils.stats import compute_form_trend, compute_score_stats


def test_compute_form_trend_insufficient_matches():
    score_list = [(1, 0)] * 8
    assert compute_form_trend(score_list) == "❓"


def test_compute_form_trend_improving():
    earlier = [(0, 1)] * 6  # all losses -> 0 points
    recent = [(2, 1)] * 3  # all wins -> 3 points each
    score_list = earlier + recent
    assert compute_form_trend(score_list) == "📈"


def test_compute_form_trend_declining():
    earlier = [(2, 1)] * 6  # all wins -> 3 points each
    recent = [(0, 1)] * 3  # all losses -> 0 points
    score_list = earlier + recent
    assert compute_form_trend(score_list) == "📉"


def test_compute_form_trend_stable():
    earlier = [(1, 1)] * 6  # all draws -> 1 point each
    recent = [(2, 2)] * 3  # all draws -> 1 point each
    score_list = earlier + recent
    assert compute_form_trend(score_list) == "➖"


def test_compute_score_stats_under_ten_matches():
    data = {
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-08",
            "2024-01-15",
            "2024-01-22",
        ]),
        "HomeTeam": ["TeamA", "TeamC", "TeamA", "TeamE"],
        "AwayTeam": ["TeamB", "TeamA", "TeamD", "TeamA"],
        "FTHG": [1, 2, 0, 1],
        "FTAG": [0, 2, 3, 1],
    }
    df = pd.DataFrame(data)

    score_list, avg_goals, variance = compute_score_stats(df, "TeamA")

    assert score_list == [(1, 0), (2, 2), (0, 3), (1, 1)]
    assert avg_goals == pytest.approx(2.5)
    assert variance == pytest.approx(1.25)
