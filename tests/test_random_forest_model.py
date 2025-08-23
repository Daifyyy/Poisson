import pandas as pd
import pytest
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils.ml.random_forest import (
    construct_features_for_match,
    predict_proba,
    load_model,
)


def _sample_df():
    data = [
        {"Date": "2024-01-01", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        {"Date": "2024-01-05", "HomeTeam": "B", "AwayTeam": "A", "FTHG": 2, "FTAG": 2, "FTR": "D"},
        {"Date": "2024-01-10", "HomeTeam": "A", "AwayTeam": "C", "FTHG": 0, "FTAG": 1, "FTR": "A"},
        {"Date": "2024-01-15", "HomeTeam": "C", "AwayTeam": "A", "FTHG": 0, "FTAG": 3, "FTR": "A"},
        {"Date": "2024-01-20", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 1, "FTR": "H"},
    ]
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_construct_features():
    df = _sample_df()
    elo_dict = {"A": 1600, "B": 1500}
    feats = construct_features_for_match(df, "A", "B", elo_dict)
    assert feats["home_recent_form"] == pytest.approx(0.4)
    assert feats["away_recent_form"] == pytest.approx(-2/3)
    assert feats["elo_diff"] == 100
    assert feats["xg_diff"] == pytest.approx(0.6)
    assert feats["home_conceded"] == pytest.approx(0.8)
    assert feats["away_conceded"] == pytest.approx(5 / 3)
    assert feats["conceded_diff"] == pytest.approx(-13 / 15)  # -0.8666...
    assert feats["home_advantage"] == 1.0
    assert feats["days_since_last_match"] == 0
    assert feats["attack_strength_diff"] == pytest.approx(0.5)
    assert feats["defense_strength_diff"] == pytest.approx(-11 / 12)


def test_predict_proba_deterministic():
    df = _sample_df()
    elo_dict = {"A": 1600, "B": 1500}
    feats = construct_features_for_match(df, "A", "B", elo_dict)
    model_data = load_model()
    probs = predict_proba(feats, model_data=model_data)
    assert probs["Home Win"] == pytest.approx(73.6666667)
    assert probs["Draw"] == pytest.approx(20.0)
    assert probs["Away Win"] == pytest.approx(6.3333333)
    assert sum(probs.values()) == pytest.approx(100.0)
