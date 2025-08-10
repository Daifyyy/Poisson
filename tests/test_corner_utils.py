import math
import pandas as pd

from utils.poisson_utils import expected_corners, poisson_corner_matrix, corner_over_under_prob


def _sample_df():
    data = [
        {"Date": "2023-08-01", "HomeTeam": "A", "AwayTeam": "B", "HC": 5, "AC": 3},
        {"Date": "2023-08-08", "HomeTeam": "A", "AwayTeam": "C", "HC": 7, "AC": 4},
        {"Date": "2023-08-15", "HomeTeam": "B", "AwayTeam": "A", "HC": 6, "AC": 2},
        {"Date": "2023-08-22", "HomeTeam": "C", "AwayTeam": "A", "HC": 8, "AC": 5},
    ]
    return pd.DataFrame(data)


def test_expected_corners():
    df = _sample_df()
    home, away = expected_corners(df, "A", "B")
    assert math.isclose(home, 5.5, rel_tol=1e-4)
    assert math.isclose(away, 3.25, rel_tol=1e-4)


def test_corner_over_under_probabilities_sum_to_100():
    matrix = poisson_corner_matrix(5.5, 3.25)
    probs = corner_over_under_prob(matrix, 8.5)
    assert math.isclose(sum(probs.values()), 100.0, abs_tol=0.1)
