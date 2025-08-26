import math
import pandas as pd

from utils.poisson_utils import calculate_match_style_score_per_match
from utils.utils_warnings import (
    calculate_match_style_score_per_match as warnings_match_style_score,
)
from utils.poisson_utils import get_team_style_vs_opponent_type


def test_calculate_match_style_score_handles_zero_shots_on_target():
    df = pd.DataFrame({
        "Date": ["2023-08-01"],
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "FTHG": pd.Series([1], dtype="Int64"),
        "FTAG": pd.Series([1], dtype="Int64"),
        "HS": pd.Series([0], dtype="Int64"),
        "AS": pd.Series([0], dtype="Int64"),
        "HST": pd.Series([0], dtype="Int64"),
        "AST": pd.Series([0], dtype="Int64"),
        "HC": pd.Series([0], dtype="Int64"),
        "AC": pd.Series([0], dtype="Int64"),
        "HY": pd.Series([0], dtype="Int64"),
        "AY": pd.Series([0], dtype="Int64"),
        "HR": pd.Series([0], dtype="Int64"),
        "AR": pd.Series([0], dtype="Int64"),
        "HF": pd.Series([0], dtype="Int64"),
        "AF": pd.Series([0], dtype="Int64"),
    })

    result = calculate_match_style_score_per_match(df)
    assert math.isclose(result["Konverze"].iloc[0], 20.0, rel_tol=1e-9)


def test_warning_match_style_score_handles_zero_shots_on_target():
    df = pd.DataFrame({
        "Date": ["2023-08-01"],
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
        "FTHG": pd.Series([1], dtype="Int64"),
        "FTAG": pd.Series([1], dtype="Int64"),
        "HS": pd.Series([0], dtype="Int64"),
        "AS": pd.Series([0], dtype="Int64"),
        "HST": pd.Series([0], dtype="Int64"),
        "AST": pd.Series([0], dtype="Int64"),
        "HC": pd.Series([0], dtype="Int64"),
        "AC": pd.Series([0], dtype="Int64"),
        "HY": pd.Series([0], dtype="Int64"),
        "AY": pd.Series([0], dtype="Int64"),
        "HR": pd.Series([0], dtype="Int64"),
        "AR": pd.Series([0], dtype="Int64"),
        "HF": pd.Series([0], dtype="Int64"),
        "AF": pd.Series([0], dtype="Int64"),
    })

    result = warnings_match_style_score(df)
    assert math.isclose(result["Konverze"].iloc[0], 20.0, rel_tol=1e-9)


def test_get_team_style_vs_opponent_type_returns_mapping():
    df = pd.DataFrame(
        {
            "Date": ["2023-08-01", "2023-08-08"],
            "HomeTeam": ["A", "B"],
            "AwayTeam": ["B", "A"],
            "FTHG": pd.Series([1, 0], dtype="Int64"),
            "FTAG": pd.Series([0, 2], dtype="Int64"),
            "HS": pd.Series([10, 8], dtype="Int64"),
            "AS": pd.Series([5, 7], dtype="Int64"),
            "HST": pd.Series([4, 3], dtype="Int64"),
            "AST": pd.Series([2, 4], dtype="Int64"),
            "HC": pd.Series([3, 1], dtype="Int64"),
            "AC": pd.Series([1, 2], dtype="Int64"),
            "HY": pd.Series([1, 2], dtype="Int64"),
            "AY": pd.Series([2, 1], dtype="Int64"),
            "HR": pd.Series([0, 0], dtype="Int64"),
            "AR": pd.Series([0, 0], dtype="Int64"),
            "HF": pd.Series([10, 12], dtype="Int64"),
            "AF": pd.Series([11, 9], dtype="Int64"),
        }
    )
    result = get_team_style_vs_opponent_type(df, "A", "B")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"Tempo", "Góly", "Konverze", "Agrese"}
    assert all(0 <= v <= 100 for v in result.values())
