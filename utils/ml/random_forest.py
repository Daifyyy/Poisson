"""Random Forest model for football match outcome prediction.

This module trains a ``RandomForestClassifier`` on historical match data
stored in CSV files under ``data/*_combined_full_updated.csv``. It extracts
several pre-match numeric features and encodes the categorical full-time
result (``FTR``) as labels.

Expected feature schema used for both training and prediction:
    - ``home_recent_form``: rolling average of home team's results over last
      5 matches (1 win, 0 draw, -1 loss).
    - ``away_recent_form``: rolling average of away team's results over last
      5 matches.
    - ``elo_diff``: current ELO rating difference (home - away) before the
      match.
    - ``xg_diff``: difference in rolling average goals scored (home - away)
      over the last 5 matches, serving as a proxy for expected goals.

The public API exposes three functions:

``train_model(data_dir='data')``
    Train a model and return ``(model, feature_names, label_encoder, score)``.

``save_model(model, feature_names, label_encoder, path)``
    Persist the model with ``joblib``.

``predict_outcome(features, model_path)``
    Load the persisted model and return the predicted outcome label
    (``'H'``, ``'D'`` or ``'A'``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_MODEL_PATH = Path(__file__).with_name("random_forest_model.joblib")


def _load_matches(data_dir: str | Path) -> pd.DataFrame:
    """Load all historical CSVs and return a concatenated DataFrame."""
    data_dir = Path(data_dir)
    frames = []
    for csv in sorted(data_dir.glob("*_combined_full_updated.csv")):
        df = pd.read_csv(csv)
        # ensure consistent datetime and sort later
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No training files found in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("Date", inplace=True)
    return df


def _compute_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling recent-form features for home and away teams."""
    results = {"H": 1, "D": 0, "A": -1}
    df = df.copy()
    df["home_result"] = df["FTR"].map(results)
    df["away_result"] = -df["home_result"]

    df["home_recent_form"] = (
        df.groupby("HomeTeam")["home_result"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )
    df["away_recent_form"] = (
        df.groupby("AwayTeam")["away_result"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )
    df.drop(columns=["home_result", "away_result"], inplace=True)
    return df


def _compute_expected_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate expected goals using rolling averages of goals scored."""
    df = df.copy()
    df["home_xg"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["away_xg"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["xg_diff"] = df["home_xg"] - df["away_xg"]
    return df


def _compute_elo_difference(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Calculate ELO rating difference before each match."""
    df = df.copy()
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    ratings = {team: 1500 for team in teams}
    elo_diffs: list[float] = []

    for _, row in df.iterrows():
        home, away, result = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        elo_diffs.append(ratings[home] - ratings[away])

        # update ratings after match
        res_home = 1 if result == "H" else 0.5 if result == "D" else 0
        res_away = 1 - res_home
        exp_home = 1 / (1 + 10 ** ((ratings[away] - ratings[home]) / 400))
        exp_away = 1 - exp_home
        ratings[home] += k * (res_home - exp_home)
        ratings[away] += k * (res_away - exp_away)

    df["elo_diff"] = elo_diffs
    return df


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Iterable[str], LabelEncoder]:
    """Return feature matrix X, labels y and metadata."""
    df = _compute_recent_form(df)
    df = _compute_expected_goals(df)
    df = _compute_elo_difference(df)

    features = ["home_recent_form", "away_recent_form", "elo_diff", "xg_diff"]
    X = df[features]
    y_raw = df["FTR"].astype(str)

    mask = X.notna().all(axis=1) & y_raw.notna()
    X = X[mask]
    y_raw = y_raw[mask]

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)
    return X, y, features, label_enc


def train_model(data_dir: str | Path = "data") -> Tuple[RandomForestClassifier, Iterable[str], LabelEncoder, float]:
    """Train a ``RandomForestClassifier`` on historical data.

    Parameters
    ----------
    data_dir:
        Directory containing historical CSV files.

    Returns
    -------
    model:
        Trained ``RandomForestClassifier``.
    feature_names:
        Iterable of feature column names used for training.
    label_encoder:
        Encoder translating between labels and ``'H'``/``'D'``/``'A'``.
    score:
        Validation accuracy on the hold-out set.
    """
    df = _load_matches(data_dir)
    X, y, feature_names, label_enc = _prepare_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    return model, feature_names, label_enc, score


def save_model(
    model: RandomForestClassifier,
    feature_names: Iterable[str],
    label_encoder: LabelEncoder,
    path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Persist model, feature names and label encoder using ``joblib``."""
    joblib.dump(
        {"model": model, "feature_names": list(feature_names), "label_encoder": label_encoder},
        Path(path),
    )


def _load_model(path: str | Path = DEFAULT_MODEL_PATH):
    data = joblib.load(Path(path))
    return data["model"], data["feature_names"], data["label_encoder"]


def predict_outcome(features: Dict[str, float], model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
    """Predict match outcome using a saved model.

    Parameters
    ----------
    features:
        Mapping of feature name to value. Must include the ``feature_names``
        documented above.
    model_path:
        Path to a saved model created by :func:`save_model`.

    Returns
    -------
    str
        Predicted full-time result label: ``'H'`` (home win), ``'D'`` (draw) or
        ``'A'`` (away win).
    """
    model, feature_names, label_enc = _load_model(model_path)
    X = pd.DataFrame([features], columns=feature_names)
    pred = model.predict(X)[0]
    return label_enc.inverse_transform([pred])[0]


__all__ = ["train_model", "save_model", "predict_outcome"]
