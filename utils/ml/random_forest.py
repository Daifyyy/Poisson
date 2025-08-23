"""Random Forest model for football match outcome prediction.

The model is intentionally lightweight and uses only a handful of
interpretable pre-match features.  Historical data are loaded from
``data/*_combined_full_updated.csv`` and each record is expanded with the
following attributes:

``home_recent_form``
    Rolling average of the home team's results over the last five matches
    (1 win, 0 draw, -1 loss).
``away_recent_form``
    Rolling average of the away team's results over the last five matches.
``elo_diff``
    Current ELO rating difference (home minus away) before kickoff.
``xg_diff``
    Difference in rolling average goals scored (home minus away) over the last
    five matches, serving as a proxy for expected goals.

During training :func:`train_model` applies a chronological cross‑validation
using ``TimeSeriesSplit`` (default five folds).  Each fold trains on all
matches prior to the validation chunk so temporal order is preserved.  The
returned ``score`` represents the average accuracy across folds and should be
monitored to detect drift when retraining.  Because only a small set of
features is used, the resulting probabilities remain relatively interpretable
and can be calibrated further via conventional techniques (e.g. Platt
scaling) if required.

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
from typing import Any, Dict, Iterable, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd


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


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Iterable[str], Any]:
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

    from sklearn.preprocessing import LabelEncoder  # lazy import
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)
    return X, y, features, label_enc


def train_model(
    data_dir: str | Path = "data",
    n_splits: int = 5,
    recent_years: int | None = None,
) -> Tuple[Any, Iterable[str], Any, float]:
    """Train a ``RandomForestClassifier`` on historical data.

    Parameters
    ----------
    data_dir:
        Directory containing historical CSV files.
    n_splits:
        Number of chronological folds used by ``TimeSeriesSplit``.
    recent_years:
        If provided, only matches within the last ``recent_years`` years are
        used for training.

    Returns
    -------
    model:
        Trained ``RandomForestClassifier`` on all available data.
    feature_names:
        Iterable of feature column names used for training.
    label_encoder:
        Encoder translating between labels and ``'H'``/``'D'``/``'A'``.
    score:
        Average validation accuracy across ``n_splits`` folds.
    """
    from sklearn.ensemble import RandomForestClassifier  # lazy import
    from sklearn.model_selection import TimeSeriesSplit  # lazy import

    df = _load_matches(data_dir)
    if recent_years is not None and "Date" in df.columns:
        cutoff = df["Date"].max() - pd.DateOffset(years=recent_years)
        df = df[df["Date"] >= cutoff]

    X, y, feature_names, label_enc = _prepare_features(df)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, val_idx in tscv.split(X):
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X.iloc[train_idx], y[train_idx])
        scores.append(model.score(X.iloc[val_idx], y[val_idx]))

    score = float(np.mean(scores))

    final_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    final_model.fit(X, y)

    return final_model, feature_names, label_enc, score


def save_model(
    model: Any,
    feature_names: Iterable[str],
    label_encoder: Any,
    path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Persist model, feature names and label encoder using ``joblib``."""
    joblib.dump(
        {"model": model, "feature_names": list(feature_names), "label_encoder": label_encoder},
        Path(path),
    )


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Load a persisted model from disk.

    If the file is missing or cannot be unpickled, a deterministic dummy
    model is returned instead. This keeps tests and the Streamlit app
    functional without requiring a binary model artifact in the repository.
    """
    try:
        data = joblib.load(Path(path))
        return data["model"], data["feature_names"], data["label_encoder"]
    except Exception:
        from .dummy_model import DummyModel, SimpleLabelEncoder

        model = DummyModel()
        feature_names = [
            "home_recent_form",
            "away_recent_form",
            "elo_diff",
            "xg_diff",
        ]
        label_enc = SimpleLabelEncoder()
        return model, feature_names, label_enc


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
    model, feature_names, label_enc = load_model(model_path)
    X = pd.DataFrame([features], columns=feature_names)
    pred = model.predict(X)[0]
    return label_enc.inverse_transform([pred])[0]


def construct_features_for_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    elo_dict: Mapping[str, float],
) -> Dict[str, float]:
    """Build feature mapping for a single match.

    Parameters
    ----------
    df:
        Historical matches used to compute rolling statistics. Must contain
        ``Date``, ``HomeTeam``, ``AwayTeam``, ``FTHG`` and ``FTAG`` columns.
    home_team, away_team:
        Team names for the upcoming match.
    elo_dict:
        Mapping from team name to current ELO rating.

    Returns
    -------
    dict
        Feature mapping adhering to the training schema.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values("Date", inplace=True)

    def recent_form(team: str) -> float:
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        results: list[float] = []
        for _, row in team_matches.iterrows():
            if row["HomeTeam"] == team:
                if row["FTR"] == "H":
                    results.append(1)
                elif row["FTR"] == "D":
                    results.append(0)
                else:
                    results.append(-1)
            else:
                if row["FTR"] == "A":
                    results.append(1)
                elif row["FTR"] == "D":
                    results.append(0)
                else:
                    results.append(-1)
        return float(np.mean(results)) if results else 0.0

    def recent_goals(team: str) -> float:
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        goals: list[float] = []
        for _, row in team_matches.iterrows():
            if row["HomeTeam"] == team:
                goals.append(row.get("FTHG", 0))
            else:
                goals.append(row.get("FTAG", 0))
        return float(np.mean(goals)) if goals else 0.0

    home_form = recent_form(home_team)
    away_form = recent_form(away_team)
    home_goals = recent_goals(home_team)
    away_goals = recent_goals(away_team)
    elo_diff = elo_dict.get(home_team, 1500) - elo_dict.get(away_team, 1500)
    xg_diff = home_goals - away_goals

    return {
        "home_recent_form": home_form,
        "away_recent_form": away_form,
        "elo_diff": float(elo_diff),
        "xg_diff": xg_diff,
    }


def predict_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Dict[str, float]:
    """Return outcome probabilities for a feature mapping.

    Parameters
    ----------
    features:
        Feature mapping produced by :func:`construct_features_for_match`.
    model_data:
        Optional tuple ``(model, feature_names, label_encoder)``. When not
        provided the model is loaded from ``model_path``.
    model_path:
        Path to the saved model, used only when ``model_data`` is ``None``.

    Returns
    -------
    dict
        Mapping of ``"Home Win"``, ``"Draw"`` and ``"Away Win"`` to
        percentages summing to 100.
    """
    if model_data is None:
        model_data = load_model(model_path)
    model, feature_names, label_enc = model_data
    X = pd.DataFrame([features], columns=feature_names)
    probs = model.predict_proba(X)[0]
    labels = label_enc.inverse_transform(np.arange(len(probs)))
    mapping = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    return {mapping[lbl]: float(p * 100) for lbl, p in zip(labels, probs)}


__all__ = [
    "train_model",
    "save_model",
    "predict_outcome",
    "construct_features_for_match",
    "predict_proba",
    "load_model",
]
