import pathlib, sys
import numpy as np
import pandas as pd
import pytest

# aby šel importovat balíček utils z repo kořene
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from sklearn.utils.class_weight import compute_class_weight  # <-- chybějící import

from utils.ml.random_forest import (
    construct_features_for_match,
    predict_proba,
    load_model,
    load_over25_model,
    train_model,
    predict_over25_proba,
    _prepare_features,
)


def _sample_df():
    data = [
        {"Date": "2024-01-01","HomeTeam": "A","AwayTeam": "B","FTHG": 1,"FTAG": 0,"FTR": "H","HS": 5,"AS": 4,"HST": 3,"AST": 2,"HC": 4,"AC": 3,},
        {"Date": "2024-01-05","HomeTeam": "B","AwayTeam": "A","FTHG": 2,"FTAG": 2,"FTR": "D","HS": 7,"AS": 6,"HST": 4,"AST": 3,"HC": 5,"AC": 4,},
        {"Date": "2024-01-10","HomeTeam": "A","AwayTeam": "C","FTHG": 0,"FTAG": 1,"FTR": "A","HS": 4,"AS": 5,"HST": 2,"AST": 3,"HC": 3,"AC": 4,},
        {"Date": "2024-01-15","HomeTeam": "C","AwayTeam": "A","FTHG": 0,"FTAG": 3,"FTR": "A","HS": 6,"AS": 8,"HST": 1,"AST": 5,"HC": 2,"AC": 6,},
        {"Date": "2024-01-20","HomeTeam": "A","AwayTeam": "B","FTHG": 2,"FTAG": 1,"FTR": "H","HS": 8,"AS": 7,"HST": 5,"AST": 3,"HC": 6,"AC": 5,},
    ]
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_construct_features():
    df = _sample_df()
    elo_dict = {"A": 1600, "B": 1500}
    feats = construct_features_for_match(df, "A", "B", elo_dict)
    assert feats["home_recent_form"] == pytest.approx(0.4)
    assert feats["away_recent_form"] == pytest.approx(-2 / 3)
    assert feats["elo_diff"] == 100
    assert feats["xg_diff"] == pytest.approx(0.6)
    assert feats["home_conceded"] == pytest.approx(0.8)
    assert feats["away_conceded"] == pytest.approx(5 / 3)
    assert feats["conceded_diff"] == pytest.approx(-13 / 15)
    assert feats["home_advantage"] == 1.0
    assert feats["days_since_last_match"] == 0
    assert feats["attack_strength_diff"] == pytest.approx(0.5)
    assert feats["defense_strength_diff"] == pytest.approx(-11 / 12)


def test_predict_proba_deterministic():
    df = _sample_df()
    elo_dict = {"A": 1600, "B": 1500}
    feats = construct_features_for_match(df, "A", "B", elo_dict)
    model_data = load_model()
    model = model_data[0]
    assert type(model).__name__ != "DummyModel"
    probs = predict_proba(feats, model_data=model_data, alpha=0.1)
    assert sum(probs.values()) == pytest.approx(100.0)
    for p in probs.values():
        assert 0 <= p <= 100


def test_predict_over25_proba_accepts_alpha():
    df = _sample_df()
    elo_dict = {"A": 1600, "B": 1500}
    feats = construct_features_for_match(df, "A", "B", elo_dict)
    model_data = load_over25_model()
    prob = predict_over25_proba(feats, model_data=model_data, alpha=0.1)
    assert 0 <= prob <= 100


def test_train_model_applies_class_weight(tmp_path):
    df = pd.concat([_sample_df()] * 3, ignore_index=True)
    df["Date"] = df["Date"] + pd.to_timedelta(df.index, unit="D")
    csv_path = tmp_path / "sample_combined_full_updated.csv"
    df.to_csv(csv_path, index=False)

    model, *_ = train_model(
        data_dir=tmp_path,
        n_splits=2,
        n_iter=1,
        max_samples=50,
        decay_factor=0.01,
        balance_classes=True,
    )
    assert model is not None

    _, y, _, _ = _prepare_features(df)
    classes = np.unique(y)
    expected_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    model_weights = model.estimator.named_steps["model"].class_weight
    for cls, weight in zip(classes, expected_weights):
        assert model_weights[cls] == pytest.approx(weight)


def test_train_model_no_class_weight_by_default(tmp_path):
    """Bez explicitního balance_classes by se class_weight neměl nastavovat."""
    df = pd.concat([_sample_df()] * 3, ignore_index=True)
    df["Date"] = df["Date"] + pd.to_timedelta(df.index, unit="D")
    csv_path = tmp_path / "sample_combined_full_updated.csv"
    df.to_csv(csv_path, index=False)

    model, *_ = train_model(
        data_dir=tmp_path,
        n_splits=2,
        n_iter=1,
        max_samples=50,
        decay_factor=0.01,
        # balance_classes nezadáváme
    )
    assert model.estimator.named_steps["model"].class_weight is None


def test_train_model_accepts_param_distributions(tmp_path):
    """Kontrola, že RandomizedSearch převezme pevné parametry."""
    df = pd.concat([_sample_df()] * 3, ignore_index=True)
    df["Date"] = df["Date"] + pd.to_timedelta(df.index, unit="D")
    csv_path = tmp_path / "sample_combined_full_updated.csv"
    df.to_csv(csv_path, index=False)

    custom = {"model__n_estimators": [10], "model__max_depth": [5]}
    # train_model vrací best_params jako pátou položku
    _, _, _, _, params, _ = train_model(
        data_dir=tmp_path,
        n_splits=2,
        n_iter=1,
        max_samples=50,
        param_distributions=custom,
    )
    assert params["model__n_estimators"] == 10
    assert params["model__max_depth"] == 5
