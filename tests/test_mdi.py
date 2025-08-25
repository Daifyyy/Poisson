import pandas as pd
import pytest

from utils.poisson_utils.mdi import calculate_mdi


def test_calculate_mdi_range():
    row = pd.Series(
        {
            "HS": 30,
            "AS": 0,
            "HST": 20,
            "AST": 0,
            "HC": 15,
            "AC": 0,
            "HF": 0,
            "AF": 30,
            "HY": 0,
            "AY": 10,
            "HR": 0,
            "AR": 5,
        }
    )
    league_avgs = {
        "HS": 10,
        "AS": 10,
        "HST": 5,
        "AST": 5,
        "HC": 5,
        "AC": 5,
        "HF": 10,
        "AF": 10,
        "HY": 5,
        "AY": 5,
        "HR": 1,
        "AR": 1,
    }
    val = calculate_mdi(row, league_avgs, opponent_strength=2.0)
    assert 0.0 <= val <= 100.0


def test_stronger_opponent_produces_higher_mdi():
    row = pd.Series(
        {
            "HS": 10,
            "AS": 5,
            "HST": 5,
            "AST": 2,
            "HC": 3,
            "AC": 1,
            "HF": 8,
            "AF": 10,
            "HY": 2,
            "AY": 4,
            "HR": 0,
            "AR": 1,
        }
    )
    league_avgs = {
        "HS": 10,
        "AS": 10,
        "HST": 5,
        "AST": 5,
        "HC": 3,
        "AC": 3,
        "HF": 9,
        "AF": 9,
        "HY": 3,
        "AY": 3,
        "HR": 1,
        "AR": 1,
    }
    weak = calculate_mdi(row, league_avgs, opponent_strength=0.8)
    strong = calculate_mdi(row, league_avgs, opponent_strength=1.2)
    assert weak < strong


@pytest.mark.parametrize(
    "row",
    [
        pd.Series({}),
        pd.Series({col: 0 for col in ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]}),
    ],
)
def test_calculate_mdi_default_value(row):
    assert calculate_mdi(row, {}, opponent_strength=1.0) == 50.0
