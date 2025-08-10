import math

from utils.poisson_utils import poisson_prediction_matrix, over_under_prob


def test_over_under_probabilities_sum_to_100():
    matrix = poisson_prediction_matrix(1.3, 0.9)
    for threshold in (1.5, 2.5, 3.5):
        probs = over_under_prob(matrix, threshold)
        total = sum(probs.values())
        assert math.isclose(total, 100.0, abs_tol=0.1)

