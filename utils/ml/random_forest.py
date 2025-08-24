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
import logging


logger = logging.getLogger(__name__)
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
    """Approximate expected goals using rolling averages of scored and conceded goals."""
    df = df.copy()
    df["home_xg"] = df.groupby("HomeTeam")["FTHG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_xg"] = df.groupby("AwayTeam")["FTAG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["xg_diff"] = df["home_xg"] - df["away_xg"]

    df["home_conceded"] = df.groupby("HomeTeam")["FTAG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_conceded"] = df.groupby("AwayTeam")["FTHG"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["conceded_diff"] = df["home_conceded"] - df["away_conceded"]
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


def _clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme feature values to reasonable ranges."""
    bounds = {
        "home_recent_form": (-1, 1),
        "away_recent_form": (-1, 1),
        "elo_diff": (-400, 400),
        "xg_diff": (-5, 5),
        "home_conceded": (0, 5),
        "away_conceded": (0, 5),
        "conceded_diff": (-5, 5),
        "days_since_last_match": (-30, 30),
        "attack_strength_diff": (-2, 2),
        "defense_strength_diff": (-2, 2),
    }
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Iterable[str], Any]:
    """Return feature matrix X, labels y and metadata."""
    df = _compute_recent_form(df)
    df = _compute_expected_goals(df)
    df = _compute_elo_difference(df)
    df["home_last"] = df.groupby("HomeTeam")["Date"].shift()
    df["away_last"] = df.groupby("AwayTeam")["Date"].shift()
    df["home_rest"] = (df["Date"] - df["home_last"]).dt.days
    df["away_rest"] = (df["Date"] - df["away_last"]).dt.days
    df["days_since_last_match"] = df["home_rest"] - df["away_rest"]
    df.drop(columns=["home_last", "away_last", "home_rest", "away_rest"], inplace=True)

    from utils.poisson_utils.stats import calculate_team_strengths

    attack_strength, defense_strength, _ = calculate_team_strengths(df)
    df["attack_strength_diff"] = df["HomeTeam"].map(attack_strength) - df["AwayTeam"].map(attack_strength)
    df["defense_strength_diff"] = df["HomeTeam"].map(defense_strength) - df["AwayTeam"].map(defense_strength)
    df["home_advantage"] = 1.0

    features = [
        "home_recent_form",
        "away_recent_form",
        "elo_diff",
        "xg_diff",
        "home_conceded",
        "away_conceded",
        "conceded_diff",
        "home_advantage",
        "days_since_last_match",
        "attack_strength_diff",
        "defense_strength_diff",
    ]
    X = _clip_features(df[features])
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
) -> Tuple[Any, Iterable[str], Any, float, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train a ``RandomForestClassifier`` on historical data using
    ``RandomizedSearchCV``.

    The function mitigates class imbalance via oversampling (if ``imblearn`` is
    available) or via explicit class weights. Returns a calibrated model and
    per-class precision/recall.
    """
    from sklearn.ensemble import RandomForestClassifier  # lazy import
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_predict
    from sklearn.calibration import CalibratedClassifierCV  # lazy import
    from sklearn.metrics import precision_recall_fscore_support  # lazy import
    from sklearn.utils.class_weight import compute_class_weight  # lazy import

    df = _load_matches(data_dir)
    if recent_years is not None and "Date" in df.columns:
        cutoff = df["Date"].max() - pd.DateOffset(years=recent_years)
        df = df[df["Date"] >= cutoff]

    X, y, feature_names, label_enc = _prepare_features(df)

    # --- Handle class imbalance inside Pipeline --------------------------------
    try:  # pragma: no cover - optional dependency
        from imblearn.over_sampling import RandomOverSampler  # type: ignore
        from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
        from imblearn.ensemble import BalancedRandomForestClassifier  # type: ignore

        pipeline = ImbPipeline(
            steps=[
                ("sampler", RandomOverSampler(random_state=42)),
                ("model", BalancedRandomForestClassifier(random_state=42)),
            ]
        )
    except Exception:
        from sklearn.pipeline import Pipeline  # lazy import

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weight = {cls: weight for cls, weight in zip(classes, weights)}
        pipeline = Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        class_weight=class_weight, random_state=42
                    ),
                )
            ]
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    param_distributions = {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        random_state=42,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    search.fit(X, y)

    best_pipeline = search.best_estimator_
    score = float(search.best_score_)
    best_params = search.best_params_

    calibrated_model = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=tscv)
    calibrated_model.fit(X, y)

    # --- Precision/recall metrics ----------------------------------------------
    y_pred = cross_val_predict(best_pipeline, X, y, cv=tscv, n_jobs=-1)
    precisions, recalls, _, _ = precision_recall_fscore_support(y, y_pred, labels=np.unique(y))
    metrics = {
        label: {"precision": float(p), "recall": float(r)}
        for label, p, r in zip(label_enc.classes_, precisions, recalls)
    }

    return calibrated_model, feature_names, label_enc, score, best_params, metrics



def save_model(
    model: Any,
    feature_names: Iterable[str],
    label_encoder: Any,
    path: str | Path = DEFAULT_MODEL_PATH,
    best_params: Mapping[str, Any] | None = None,
) -> None:
    """Persist model, feature names, label encoder and parameters using ``joblib``."""
    joblib.dump(
        {
            "model": model,
            "feature_names": list(feature_names),
            "label_encoder": label_encoder,
            "best_params": best_params or {},
        },
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
        model = data["model"]
        logger.info("Loaded model type: %s", type(model).__name__)
        if type(model).__name__ == "DummyModel":
            logger.warning("Loaded fallback DummyModel; trained artifact missing?")
        return (
            model,
            data["feature_names"],
            data["label_encoder"],
            data.get("best_params", {}),
        )
    except Exception:
        from .dummy_model import DummyModel, SimpleLabelEncoder

        logger.warning("Falling back to DummyModel; could not load %s", path)
        model = DummyModel()
        feature_names = [
            "home_recent_form",
            "away_recent_form",
            "elo_diff",
            "xg_diff",
            "home_conceded",
            "away_conceded",
            "conceded_diff",
            "home_advantage",
            "days_since_last_match",
            "attack_strength_diff",
            "defense_strength_diff",
        ]
        label_enc = SimpleLabelEncoder()
        return model, feature_names, label_enc, {}


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
    model, feature_names, label_enc, _ = load_model(model_path)
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

    def recent_conceded(team: str) -> float:
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        goals: list[float] = []
        for _, row in team_matches.iterrows():
            if row["HomeTeam"] == team:
                goals.append(row.get("FTAG", 0))
            else:
                goals.append(row.get("FTHG", 0))
        return float(np.mean(goals)) if goals else 0.0

    def last_match_date(team: str) -> pd.Timestamp | None:
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        if team_matches.empty:
            return None
        return team_matches["Date"].max()

    home_form = recent_form(home_team)
    away_form = recent_form(away_team)
    home_goals = recent_goals(home_team)
    away_goals = recent_goals(away_team)
    home_conceded = recent_conceded(home_team)
    away_conceded = recent_conceded(away_team)
    elo_diff = elo_dict.get(home_team, 1500) - elo_dict.get(away_team, 1500)
    xg_diff = home_goals - away_goals
    conceded_diff = home_conceded - away_conceded
    home_advantage = 1.0
    home_last = last_match_date(home_team)
    away_last = last_match_date(away_team)
    days_since_last_match = (
        float((home_last - away_last).days) if home_last and away_last else 0.0
    )
    from utils.poisson_utils.stats import calculate_team_strengths

    attack_strength, defense_strength, _ = calculate_team_strengths(df)
    attack_strength_diff = attack_strength.get(home_team, 0.0) - attack_strength.get(away_team, 0.0)
    defense_strength_diff = defense_strength.get(home_team, 0.0) - defense_strength.get(away_team, 0.0)

    feats = {
        "home_recent_form": home_form,
        "away_recent_form": away_form,
        "elo_diff": float(elo_diff),
        "xg_diff": xg_diff,
        "home_conceded": home_conceded,
        "away_conceded": away_conceded,
        "conceded_diff": conceded_diff,
        "home_advantage": home_advantage,
        "days_since_last_match": days_since_last_match,
        "attack_strength_diff": attack_strength_diff,
        "defense_strength_diff": defense_strength_diff,
    }
    return _clip_features(pd.DataFrame([feats])).iloc[0].to_dict()


def predict_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any] | Tuple[Any, Iterable[str], Any, Mapping[str, Any]] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Dict[str, float]:
    """Return outcome probabilities for a feature mapping.

    Parameters
    ----------
    features:
        Feature mapping produced by :func:`construct_features_for_match`.
    model_data:
        Optional tuple ``(model, feature_names, label_encoder)`` or
        ``(model, feature_names, label_encoder, best_params)``. When not
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
    model, feature_names, label_enc = model_data[:3]
    X = _clip_features(pd.DataFrame([features], columns=feature_names))
    probs = model.predict_proba(X)[0]
    # shrink towards uniform prior to dampen extreme probabilities
    alpha = 0.15
    probs = (1 - alpha) * probs + alpha * (1.0 / len(probs))
    probs = probs / probs.sum()
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
