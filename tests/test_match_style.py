import math
import pandas as pd

from utils.poisson_utils import calculate_match_style_score_per_match


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
