"""Lightweight Random Forest helpers used by the Streamlit app.

Updated to better handle different data providers and improved error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from .dummy_model import DummyModel, SimpleLabelEncoder

DEFAULT_MODEL_PATH = Path(__file__).with_name("random_forest_model.joblib")
DEFAULT_OVER25_MODEL_PATH = Path(__file__).with_name(
    "random_forest_over25_model.joblib"
)


def load_csv_data(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load match data from local CSV files.

    The project originally fetched training data from an external API.  For
    offline use we instead read the prepared ``*_combined_full_updated.csv``
    files from ``data_dir`` and reshape them into the FBR-like format expected
    by the existing feature engineering pipeline.
    """
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []
    for csv in sorted(data_dir.glob("*_combined_full_updated.csv")):
        df = pd.read_csv(csv)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No training files found in {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    df.sort_values("Date", inplace=True)
    df = _add_elo_difference(df)

    # pseudo xG derived purely from shot counts (0.1*shots + 0.3*sot)
    shots_h = df.get("HS", pd.Series(0, index=df.index))
    shots_a = df.get("AS", pd.Series(0, index=df.index))
    sot_h = df.get("HST", pd.Series(0, index=df.index))
    sot_a = df.get("AST", pd.Series(0, index=df.index))
    xg_h = 0.1 * shots_h + 0.3 * sot_h
    xg_a = 0.1 * shots_a + 0.3 * sot_a

    # Map football-data column names to FBR-style names used internally
    out = pd.DataFrame(
        {
            "date": df["Date"],
            "team_name_H": df["HomeTeam"],
            "team_name_A": df["AwayTeam"],
            "gf_H": df["FTHG"],
            "gf_A": df["FTAG"],
            "ga_H": df["FTAG"],
            "ga_A": df["FTHG"],
            # pseudo expected goals from shots
            "xg_H": xg_h,
            "xg_A": xg_a,
            "xga_H": xg_a,
            "xga_A": xg_h,
            "shots_H": shots_h,
            "shots_A": shots_a,
            "sot_H": sot_h,
            "sot_A": sot_a,
            "FTR": df["FTR"],
            "B365H": df.get("B365H"),
            "B365D": df.get("B365D"),
            "B365A": df.get("B365A"),
            "B365>2.5": df.get("B365>2.5"),
            "B365<2.5": df.get("B365<2.5"),
            "elo_diff": df["elo_diff"],
        }
    )

    return out


def _add_elo_difference(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Compute and append ``elo_diff`` from historical results.

    Parameters mirror the simple approach from :mod:`tests.test_random_forest_model`
    where ratings start at 1500 and are updated after each match.
    """
    df = df.copy()
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    ratings: Dict[str, float] = {team: 1500.0 for team in teams}
    diffs: list[float] = []

    for _, row in df.iterrows():
        h, a, res = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        diffs.append(ratings[h] - ratings[a])
        res_home = 1 if res == "H" else 0.5 if res == "D" else 0
        exp_home = 1 / (1 + 10 ** ((ratings[a] - ratings[h]) / 400))
        ratings[h] += k * (res_home - exp_home)
        ratings[a] += k * ((1 - res_home) - (1 - exp_home))

    df["elo_diff"] = diffs
    return df


class SampleWeightPipeline:
    """Minimal pipeline that forwards ``sample_weight`` to the final estimator."""

    def __init__(self, steps):
        from sklearn.pipeline import Pipeline

        self.pipeline = Pipeline(steps)

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            last_step = self.pipeline.steps[-1][0] + "__sample_weight"
            fit_params[last_step] = sample_weight
        return self.pipeline.fit(X, y, **fit_params)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def get_params(self, *args, **kwargs):
        return self.pipeline.get_params(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        return self.pipeline.set_params(*args, **kwargs)


def _clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme feature values to reasonable ranges."""
    df = df.copy()
    bounds = {
        "home_recent_form": (-1, 1),
        "away_recent_form": (-1, 1),
        "elo_diff": (-400, 400),
        "xg_diff": (-5, 5),
        "xga_diff": (-5, 5),
        "goal_balance_diff": (-5, 5),
        "shots_diff": (-20, 20),
        "shot_target_diff": (-10, 10),
        "shot_accuracy_diff": (-1, 1),
        "conversion_rate_diff": (-1, 1),
        "def_compactness_diff": (-10, 10),
        "days_since_last_match": (-30, 30),
        "attack_strength_diff": (-3, 3),
        "defense_strength_diff": (-3, 3),
        "tempo": (0, 40),
        "style_diff": (-5, 5),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df.loc[:, col] = df[col].clip(lo, hi)
    return df


def _expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error for binary outcomes."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (prob >= bins[i]) & (prob < bins[i + 1])
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(prob[mask])
            ece += abs(acc - conf) * (np.sum(mask) / len(prob))
    return float(ece)


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Iterable[str], Any]:
    """Create feature matrix and labels from FBR-style match dataframe."""
    df = df.copy()

    required_cols = [
        "date",
        "team_name_H",
        "team_name_A",
        "gf_H",
        "gf_A",
        "ga_H",
        "ga_A",
        "xg_H",
        "xg_A",
        "xga_H",
        "xga_A",
        "shots_H",
        "shots_A",
        "sot_H",
        "sot_A",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)

    results = {"H": 1, "D": 0, "A": -1}
    result_col = "FTR" if "FTR" in df.columns else "outcome"
    if result_col not in df.columns:
        raise ValueError("DataFrame must contain 'FTR' or 'outcome'")
    df["home_result"] = df[result_col].map(results)
    df["away_result"] = -df["home_result"]
    df["home_recent_form"] = df.groupby("team_name_H")["home_result"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_recent_form"] = df.groupby("team_name_A")["away_result"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df.drop(columns=["home_result", "away_result"], inplace=True)

    df["home_xg"] = df.groupby("team_name_H")["xg_H"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["away_xg"] = df.groupby("team_name_A")["xg_A"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["xg_diff"] = df["home_xg"] - df["away_xg"]

    df["home_xga"] = df.groupby("team_name_H")["xga_H"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["away_xga"] = df.groupby("team_name_A")["xga_A"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["xga_diff"] = df["home_xga"] - df["away_xga"]

    home_gf = df.groupby("team_name_H")["gf_H"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    away_gf = df.groupby("team_name_A")["gf_A"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    home_ga = df.groupby("team_name_H")["ga_H"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    away_ga = df.groupby("team_name_A")["ga_A"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["goal_balance_diff"] = (home_gf - home_ga) - (away_gf - away_ga)
    df["style_diff"] = (home_gf + home_ga) - (away_gf + away_ga)

    df["home_shots"] = df.groupby("team_name_H")["shots_H"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["away_shots"] = df.groupby("team_name_A")["shots_A"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["shots_diff"] = df["home_shots"] - df["away_shots"]
    df["tempo"] = df["home_shots"] + df["away_shots"]

    df["home_sot"] = df.groupby("team_name_H")["sot_H"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["away_sot"] = df.groupby("team_name_A")["sot_A"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df["shot_target_diff"] = df["home_sot"] - df["away_sot"]

    # Efficiency metrics
    df["shot_accuracy_diff"] = (df["home_sot"] / df["home_shots"]).replace([np.inf, -np.inf], np.nan) - (
        df["away_sot"] / df["away_shots"]
    ).replace([np.inf, -np.inf], np.nan)

    df["conversion_rate_diff"] = (home_gf / df["home_shots"]).replace([np.inf, -np.inf], np.nan) - (
        away_gf / df["away_shots"]
    ).replace([np.inf, -np.inf], np.nan)

    home_shots_against = df.groupby("team_name_H")["shots_A"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    away_shots_against = df.groupby("team_name_A")["shots_H"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    home_def = (home_shots_against / home_ga).replace([np.inf, -np.inf], np.nan)
    away_def = (away_shots_against / away_ga).replace([np.inf, -np.inf], np.nan)
    df["def_compactness_diff"] = home_def - away_def

    df["shot_accuracy_diff"] = df["shot_accuracy_diff"].fillna(0.0)
    df["conversion_rate_diff"] = df["conversion_rate_diff"].fillna(0.0)
    df["def_compactness_diff"] = df["def_compactness_diff"].fillna(0.0)

    df.drop(
        columns=[
            "home_shots",
            "away_shots",
            "home_sot",
            "away_sot",
            "home_xg",
            "away_xg",
            "home_xga",
            "away_xga",
        ],
        inplace=True,
        errors="ignore",
    )

    df["home_last"] = df.groupby("team_name_H")["date"].shift()
    df["away_last"] = df.groupby("team_name_A")["date"].shift()
    df["home_rest"] = (df["date"] - df["home_last"]).dt.days
    df["away_rest"] = (df["date"] - df["away_last"]).dt.days
    df["days_since_last_match"] = df["home_rest"] - df["away_rest"]
    df.drop(columns=["home_last", "away_last", "home_rest", "away_rest"], inplace=True)

    from utils.poisson_utils.stats import calculate_team_strengths

    tmp = df.rename(
        columns={"team_name_H": "HomeTeam", "team_name_A": "AwayTeam", "gf_H": "FTHG", "gf_A": "FTAG"}
    )
    attack_strength, defense_strength, _ = calculate_team_strengths(tmp)
    df["attack_strength_diff"] = df["team_name_H"].map(attack_strength) - df["team_name_A"].map(attack_strength)
    df["defense_strength_diff"] = df["team_name_H"].map(defense_strength) - df["team_name_A"].map(defense_strength)

    df["home_advantage"] = 1.0

    features = [
        "home_recent_form",
        "away_recent_form",
        "elo_diff",
        "xg_diff",
        "xga_diff",
        "goal_balance_diff",
        "shots_diff",
        "shot_target_diff",
        "shot_accuracy_diff",
        "conversion_rate_diff",
        "def_compactness_diff",
        "home_advantage",
        "days_since_last_match",
        "attack_strength_diff",
        "defense_strength_diff",
        "tempo",
        "style_diff",
    ]

    X = _clip_features(df[features])
    y_raw = df[result_col].astype(str)
    mask = X.notna().all(axis=1) & y_raw.notna()
    X = X[mask]
    y_raw = y_raw[mask]

    from sklearn.preprocessing import LabelEncoder

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)
    return X, y, features, label_enc


def _time_series_cross_val_predict_proba(model, X, y, cv, n_classes, sample_weight=None):
    """Manually compute out-of-fold predicted probabilities for time series CV."""
    from sklearn.base import clone

    preds = np.full((len(X), n_classes), np.nan)
    for train_idx, test_idx in cv.split(X):
        m = clone(model)
        if sample_weight is not None:
            m.fit(X.iloc[train_idx], y[train_idx], sample_weight=sample_weight[train_idx])
        else:
            m.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = m.predict_proba(X.iloc[test_idx])
    mask = ~np.isnan(preds).any(axis=1)
    return preds[mask], mask


def train_model(
    df: pd.DataFrame,
    n_splits: int = 5,
    recent_years: int | None = None,
    n_iter: int = 20,
    max_samples: int | None = None,
    param_distributions: Mapping[str, Iterable[Any]] | None = None,
) -> Tuple[Any, Iterable[str], Any, float, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train RandomForest model for match outcome prediction using FBR API data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import precision_recall_fscore_support, log_loss

    df = df.copy()
    if recent_years is not None and "date" in df.columns:
        cutoff = df["date"].max() - pd.DateOffset(years=recent_years)
        df = df[df["date"] >= cutoff]
    if max_samples is not None:
        df = df.tail(max_samples)

    X, y, feature_names, label_enc = _prepare_features(df)

    if param_distributions is None:
        param_distributions = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    labels = np.arange(len(label_enc.classes_))

    def neg_log_loss_scorer(estimator, X_val, y_val):
        prob = estimator.predict_proba(X_val)
        return -log_loss(y_val, prob, labels=labels)

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        scoring=neg_log_loss_scorer,
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    score = float(-search.best_score_)

    cal_splits = min(3, np.min(np.bincount(y)))
    cal_splits = max(2, cal_splits)
    cal_cv = StratifiedKFold(n_splits=cal_splits, shuffle=True, random_state=42)
    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=cal_cv)
    calibrated.fit(X, y)

    probs_cv, mask_cv = _time_series_cross_val_predict_proba(
        best_model, X, y, tscv, len(label_enc.classes_)
    )
    y_valid = y[mask_cv]
    y_pred = probs_cv.argmax(axis=1)
    precisions, recalls, _, _ = precision_recall_fscore_support(
        y_valid, y_pred, labels=np.arange(len(label_enc.classes_))
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(label_enc.classes_):
        y_true_bin = (y_valid == idx).astype(int)
        prob = probs_cv[:, idx]
        brier = float(np.mean((prob - y_true_bin) ** 2))
        ece = _expected_calibration_error(y_true_bin, prob)
        per_class[label] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "brier": brier,
            "ece": ece,
        }

    freq = np.bincount(y_valid) / len(y_valid)
    frequency_ll = float(log_loss(y_valid, np.tile(freq, (len(y_valid), 1))))

    bookmaker_ll = None
    book_cols_options = [
        ("B365H", "B365D", "B365A"),
        ("b365_home", "b365_draw", "b365_away"),
    ]
    for cols in book_cols_options:
        if all(c in df.columns for c in cols):
            odds = df.loc[X.index[mask_cv], list(cols)].astype(float)
            mask_odds = odds.notna().all(axis=1)
            odds = odds[mask_odds]
            if not odds.empty:
                probs = 1 / odds
                probs = probs.div(probs.sum(axis=1), axis=0)
                y_book = y_valid[mask_odds.to_numpy()]
                bookmaker_ll = float(log_loss(y_book, probs.values))
                break

    metrics = {
        "per_class": per_class,
        "baselines": {
            "frequency_log_loss": frequency_ll,
            "bookmaker_log_loss": bookmaker_ll,
        },
    }

    return calibrated, feature_names, label_enc, score, search.best_params_, metrics


def train_over25_model(
    df: pd.DataFrame,
    n_splits: int = 5,
    recent_years: int | None = None,
    n_iter: int = 20,
    max_samples: int | None = None,
    param_distributions: Mapping[str, Iterable[Any]] | None = None,
) -> Tuple[Any, Iterable[str], Any, float, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train RandomForest model for Over/Under 2.5 prediction using FBR API data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import precision_recall_fscore_support, log_loss

    df = df.copy()
    if recent_years is not None and "date" in df.columns:
        cutoff = df["date"].max() - pd.DateOffset(years=recent_years)
        df = df[df["date"] >= cutoff]
    if max_samples is not None:
        df = df.tail(max_samples)

    df["over25"] = np.where(df["gf_H"] + df["gf_A"] > 2.5, "Over 2.5", "Under 2.5")
    X, _, feature_names, _ = _prepare_features(df)
    y_raw = df.loc[X.index, "over25"]

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)

    # sample weights favouring matches between similarly rated teams
    weights = 1 / (1 + np.abs(df.loc[X.index, "elo_diff"]).to_numpy() / 400)

    if param_distributions is None:
        param_distributions = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    labels = np.arange(len(label_enc.classes_))

    def neg_log_loss_scorer(estimator, X_val, y_val):
        prob = estimator.predict_proba(X_val)
        return -log_loss(y_val, prob, labels=labels)

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        scoring=neg_log_loss_scorer,
    )
    search.fit(X, y, sample_weight=weights)
    best_model = search.best_estimator_
    score = float(-search.best_score_)

    cal_splits = min(3, np.min(np.bincount(y)))
    cal_splits = max(2, cal_splits)
    cal_cv = StratifiedKFold(n_splits=cal_splits, shuffle=True, random_state=42)
    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=cal_cv)
    calibrated.fit(X, y, sample_weight=weights)

    probs_cv, mask_cv = _time_series_cross_val_predict_proba(
        best_model, X, y, tscv, len(label_enc.classes_), sample_weight=weights
    )
    y_valid = y[mask_cv]
    y_pred = probs_cv.argmax(axis=1)
    precisions, recalls, _, _ = precision_recall_fscore_support(
        y_valid, y_pred, labels=np.arange(len(label_enc.classes_))
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(label_enc.classes_):
        y_true_bin = (y_valid == idx).astype(int)
        prob = probs_cv[:, idx]
        brier = float(np.mean((prob - y_true_bin) ** 2))
        ece = _expected_calibration_error(y_true_bin, prob)
        per_class[label] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "brier": brier,
            "ece": ece,
        }

    freq = np.bincount(y_valid) / len(y_valid)
    frequency_ll = float(log_loss(y_valid, np.tile(freq, (len(y_valid), 1))))

    bookmaker_ll = None
    book_cols_options = [
        ("B365>2.5", "B365<2.5"),
        ("b365_over25", "b365_under25"),
    ]
    for cols in book_cols_options:
        if all(c in df.columns for c in cols):
            odds = df.loc[X.index[mask_cv], list(cols)].astype(float)
            mask_odds = odds.notna().all(axis=1)
            odds = odds[mask_odds]
            if not odds.empty:
                probs = 1 / odds
                probs = probs.div(probs.sum(axis=1), axis=0)
                y_book = y_valid[mask_odds.to_numpy()]
                bookmaker_ll = float(log_loss(y_book, probs.values))
                break

    metrics = {
        "per_class": per_class,
        "baselines": {
            "frequency_log_loss": frequency_ll,
            "bookmaker_log_loss": bookmaker_ll,
        },
    }

    return calibrated, feature_names, label_enc, score, search.best_params_, metrics


def save_model(
    model: Any,
    feature_names: Iterable[str],
    label_encoder: Any,
    path: str | Path = DEFAULT_MODEL_PATH,
    best_params: Mapping[str, Any] | None = None,
) -> None:
    """Save model, feature names and label encoder to disk."""
    joblib.dump(
        {
            "model": model,
            "feature_names": list(feature_names),
            "label_encoder": label_encoder,
            "best_params": best_params or {},
        },
        Path(path),
    )


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Tuple[Any, Iterable[str], Any]:
    """Load the outcome model from disk.

    If loading fails, a very small dummy model with a fixed feature set is
    returned so the application continues to work in a limited fashion.
    """
    try:
        data = joblib.load(Path(path))
        print(f"Successfully loaded model from {path}")
        return data["model"], data["feature_names"], data.get("label_encoder")
    except Exception as e:
        print(f"Could not load model from {path}: {e}")
        print("Falling back to dummy model")
        return DummyModel(), [
            "home_recent_form",
            "away_recent_form",
            "elo_diff",
            "xg_diff",
            "xga_diff",
            "goal_balance_diff",
            "shots_diff",
            "shot_target_diff",
            "shot_accuracy_diff",
            "conversion_rate_diff",
            "def_compactness_diff",
            "home_advantage",
            "days_since_last_match",
            "attack_strength_diff",
            "defense_strength_diff",
        ], SimpleLabelEncoder()


def load_over25_model(
    path: str | Path = DEFAULT_OVER25_MODEL_PATH,
) -> Tuple[Any, Iterable[str], Any]:
    """Load the Over/Under 2.5 model from disk or return a dummy one."""
    try:
        data = joblib.load(Path(path))
        print(f"Successfully loaded over2.5 model from {path}")
        return data["model"], data["feature_names"], data.get("label_encoder")
    except Exception as e:
        print(f"Could not load over2.5 model from {path}: {e}")
        print("Falling back to dummy model")
        return DummyModel(), [
            "home_recent_form",
            "away_recent_form",
            "elo_diff",
            "xg_diff",
            "xga_diff",
            "goal_balance_diff",
            "shots_diff",
            "shot_target_diff",
            "shot_accuracy_diff",
            "conversion_rate_diff",
            "def_compactness_diff",
            "home_advantage",
            "days_since_last_match",
            "attack_strength_diff",
            "defense_strength_diff",
        ], SimpleLabelEncoder()


def construct_features_for_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    elo_dict: Dict[str, float],
    debug: bool = False,
) -> Dict[str, float]:
    """Construct feature mapping for a single match from FBR stats.

    Updated with better error handling and debug information.
    
    Args:
        df: DataFrame with match history in FBR format
        home_team: Name of home team
        away_team: Name of away team  
        elo_dict: Dictionary mapping team names to ELO ratings
        debug: If True, print debug information
    """
    if df.empty:
        print("Warning: Empty DataFrame provided")
        return _get_default_features(home_team, away_team, elo_dict)

    df = df.copy()
    
    # Handle different date column names
    date_col = None
    for col in ["date", "Date", "DATE"]:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print("Warning: No date column found")
        return _get_default_features(home_team, away_team, elo_dict)
    
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])  # Remove rows with invalid dates
    df.sort_values(date_col, inplace=True)
    
    if debug:
        print(f"DataFrame shape: {df.shape}")
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        print(f"Columns: {list(df.columns)}")

    def recent_form(team: str, window: int = 5) -> float:
        """Calculate recent form for a team (last N matches)."""
        home_matches = df[df.get("team_name_H") == team]
        away_matches = df[df.get("team_name_A") == team]
        all_matches = pd.concat([home_matches, away_matches]).sort_values(date_col)
        recent_matches = all_matches.tail(window)
        
        if debug:
            print(f"{team} recent matches: {len(recent_matches)}")
        
        vals: list[float] = []
        for _, r in recent_matches.iterrows():
            if r.get("team_name_H") == team:
                gf, ga = r.get("gf_H", 0), r.get("ga_H", 0)
            else:
                gf, ga = r.get("gf_A", 0), r.get("ga_A", 0)
            
            if pd.isna(gf) or pd.isna(ga):
                continue
                
            vals.append(1 if gf > ga else 0 if gf == ga else -1)
        
        return float(np.mean(vals)) if vals else 0.0

    def rolling_avg(team: str, col_home: str, col_away: str, window: int = 5) -> float:
        """Calculate rolling average for a stat."""
        home_matches = df[df.get("team_name_H") == team]
        away_matches = df[df.get("team_name_A") == team]
        all_matches = pd.concat([home_matches, away_matches]).sort_values(date_col)
        recent_matches = all_matches.tail(window)
        
        vals: list[float] = []
        for _, r in recent_matches.iterrows():
            if r.get("team_name_H") == team:
                val = r.get(col_home, 0.0)
            else:
                val = r.get(col_away, 0.0)
            
            if not pd.isna(val):
                vals.append(float(val))
        
        return float(np.mean(vals)) if vals else 0.0

    def last_match_date(team: str) -> Optional[pd.Timestamp]:
        """Get date of last match for a team."""
        home_matches = df[df.get("team_name_H") == team]
        away_matches = df[df.get("team_name_A") == team]
        all_matches = pd.concat([home_matches, away_matches])
        
        if all_matches.empty:
            return None
        return all_matches[date_col].max()

    # Calculate features
    try:
        home_form = recent_form(home_team)
        away_form = recent_form(away_team)

        home_xg = rolling_avg(home_team, "xg_H", "xg_A")
        away_xg = rolling_avg(away_team, "xg_H", "xg_A")
        xg_diff = home_xg - away_xg

        home_xga = rolling_avg(home_team, "xga_H", "xga_A")
        away_xga = rolling_avg(away_team, "xga_H", "xga_A")
        xga_diff = home_xga - away_xga

        home_gf = rolling_avg(home_team, "gf_H", "gf_A")
        away_gf = rolling_avg(away_team, "gf_H", "gf_A")
        home_ga = rolling_avg(home_team, "ga_H", "ga_A")
        away_ga = rolling_avg(away_team, "ga_H", "ga_A")
        goal_balance_diff = (home_gf - home_ga) - (away_gf - away_ga)
        style_diff = (home_gf + home_ga) - (away_gf + away_ga)

        home_shots = rolling_avg(home_team, "shots_H", "shots_A")
        away_shots = rolling_avg(away_team, "shots_H", "shots_A")
        shots_diff = home_shots - away_shots
        tempo = home_shots + away_shots

        home_sot = rolling_avg(home_team, "sot_H", "sot_A")
        away_sot = rolling_avg(away_team, "sot_H", "sot_A")
        shot_target_diff = home_sot - away_sot

        shot_accuracy_diff = (
            (home_sot / home_shots) if home_shots else 0.0
        ) - ((away_sot / away_shots) if away_shots else 0.0)

        conversion_rate_diff = (
            (home_gf / home_shots) if home_shots else 0.0
        ) - ((away_gf / away_shots) if away_shots else 0.0)

        home_shots_against = rolling_avg(home_team, "shots_A", "shots_H")
        away_shots_against = rolling_avg(away_team, "shots_A", "shots_H")
        def_compactness_diff = (
            (home_shots_against / home_ga) if home_ga else 0.0
        ) - ((away_shots_against / away_ga) if away_ga else 0.0)

        h_last = last_match_date(home_team)
        a_last = last_match_date(away_team)
        days_since_last = float((h_last - a_last).days) if h_last and a_last else 0.0

        # Try to calculate attack/defense strength
        attack_strength_diff = 0.0
        defense_strength_diff = 0.0
        try:
            from utils.poisson_utils.stats import calculate_team_strengths

            required_cols = [date_col, "team_name_H", "team_name_A", "gf_H", "gf_A"]
            if all(col in df.columns for col in required_cols):
                tmp = df[required_cols].rename(
                    columns={
                        date_col: "Date",
                        "team_name_H": "HomeTeam", 
                        "team_name_A": "AwayTeam",
                        "gf_H": "FTHG",
                        "gf_A": "FTAG",
                    }
                )
                atk, dfn, _ = calculate_team_strengths(tmp)
                attack_strength_diff = atk.get(home_team, 0.0) - atk.get(away_team, 0.0)
                defense_strength_diff = dfn.get(home_team, 0.0) - dfn.get(away_team, 0.0)
        except Exception as e:
            if debug:
                print(f"Could not calculate team strengths: {e}")

        features = {
            "home_recent_form": home_form,
            "away_recent_form": away_form,
            "elo_diff": float(elo_dict.get(home_team, 1500) - elo_dict.get(away_team, 1500)),
            "xg_diff": xg_diff,
            "xga_diff": xga_diff,
            "goal_balance_diff": goal_balance_diff,
            "shots_diff": shots_diff,
            "tempo": tempo,
            "shot_target_diff": shot_target_diff,
            "shot_accuracy_diff": shot_accuracy_diff,
            "conversion_rate_diff": conversion_rate_diff,
            "def_compactness_diff": def_compactness_diff,
            "home_advantage": 1.0,
            "days_since_last_match": days_since_last,
            "attack_strength_diff": attack_strength_diff,
            "defense_strength_diff": defense_strength_diff,
            "style_diff": style_diff,
        }
        
        if debug:
            print("Calculated features:")
            for k, v in features.items():
                print(f"  {k}: {v:.3f}")
        
        return features
        
    except Exception as e:
        print(f"Error calculating features: {e}")
        return _get_default_features(home_team, away_team, elo_dict)


def _get_default_features(home_team: str, away_team: str, elo_dict: Dict[str, float]) -> Dict[str, float]:
    """Return default features when calculation fails."""
    return {
        "home_recent_form": 0.0,
        "away_recent_form": 0.0,
        "elo_diff": float(elo_dict.get(home_team, 1500) - elo_dict.get(away_team, 1500)),
        "xg_diff": 0.0,
        "xga_diff": 0.0,
        "goal_balance_diff": 0.0,
        "shots_diff": 0.0,
        "tempo": 0.0,
        "shot_target_diff": 0.0,
        "shot_accuracy_diff": 0.0,
        "conversion_rate_diff": 0.0,
        "def_compactness_diff": 0.0,
        "home_advantage": 1.0,
        "days_since_last_match": 0.0,
        "attack_strength_diff": 0.0,
        "defense_strength_diff": 0.0,
        "style_diff": 0.0,
    }


def predict_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    alpha: float = 0.15,
    debug: bool = False,
) -> Dict[str, float]:
    """Return outcome probabilities (Home/Draw/Away) in percent."""

    if model_data is None:
        model_data = load_model(model_path)
    
    model, feature_names, label_enc = model_data
    
    # Ensure all required features are present
    feature_vector = []
    for fname in feature_names:
        val = features.get(fname, 0.0)
        if pd.isna(val):
            val = 0.0
        feature_vector.append(val)
    
    if debug:
        print(f"Feature vector length: {len(feature_vector)}")
        print(f"Expected features: {len(feature_names)}")
    
    X = np.array([feature_vector])
    
    try:
        probs = model.predict_proba(X)[0]
        # Apply smoothing
        probs = (1 - alpha) * probs + alpha * (1.0 / len(probs))
        
        # Map to readable labels
        if hasattr(label_enc, 'inverse_transform'):
            labels = label_enc.inverse_transform(np.arange(len(probs)))
        else:
            labels = ["H", "D", "A"]  # Default labels
            
        mapping = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
        result = {mapping.get(lbl, lbl): float(p * 100) for lbl, p in zip(labels, probs)}
        
        if debug:
            print(f"Prediction probabilities: {result}")
            
        return result
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        # Return default probabilities
        return {"Home Win": 33.33, "Draw": 33.33, "Away Win": 33.34}


def predict_over25_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any] | None = None,
    model_path: str | Path = DEFAULT_OVER25_MODEL_PATH,
    alpha: float = 0.05,
    debug: bool = False,
) -> float:
    """Return probability (0-100) that a match finishes Over 2.5 goals."""

    if model_data is None:
        model_data = load_over25_model(model_path)
    
    model, feature_names, label_enc = model_data
    
    # Ensure all required features are present
    feature_vector = []
    for fname in feature_names:
        val = features.get(fname, 0.0)
        if pd.isna(val):
            val = 0.0
        feature_vector.append(val)
    
    X = np.array([feature_vector])
    
    try:
        raw_proba = model.predict_proba(X)[0]

        if label_enc is not None and hasattr(label_enc, 'classes_'):
            expected = np.arange(len(label_enc.classes_))
            model_classes = getattr(model, "classes_", expected)
            probs_full = np.zeros(len(expected))
            
            for p, cls in zip(raw_proba, model_classes):
                if int(cls) < len(probs_full):
                    probs_full[int(cls)] = p
                    
            classes = label_enc.inverse_transform(np.arange(len(expected)))
            
            try:
                over_idx = list(classes).index("Over 2.5")
                prob = probs_full[over_idx]
            except ValueError:
                # If "Over 2.5" not found, assume binary classification
                prob = raw_proba[1] if len(raw_proba) > 1 else 0.5
        else:
            # Binary classification without label encoder
            prob = raw_proba[1] if len(raw_proba) > 1 else 0.5

        # Apply smoothing
        prob = (1 - alpha) * prob + alpha * 0.5
        
        if debug:
            print(f"Over 2.5 probability: {prob * 100:.2f}%")
            
        return float(prob * 100)
        
    except Exception as e:
        print(f"Error predicting over 2.5: {e}")
        return 50.0  # Default 50% probability


def validate_dataframe(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate that DataFrame has required columns for feature construction."""
    
    required_cols = {
        "team_name_H": "team_name_H" in df.columns,
        "team_name_A": "team_name_A" in df.columns,
        "gf_H": "gf_H" in df.columns,
        "gf_A": "gf_A" in df.columns,
        "ga_H": "ga_H" in df.columns,
        "ga_A": "ga_A" in df.columns,
        "date": any(col in df.columns for col in ["date", "Date", "DATE"]),
    }
    
    optional_cols = {
        "xg_H": "xg_H" in df.columns,
        "xg_A": "xg_A" in df.columns,
        "xga_H": "xga_H" in df.columns,
        "xga_A": "xga_A" in df.columns,
        "shots_H": "shots_H" in df.columns,
        "shots_A": "shots_A" in df.columns,
        "sot_H": "sot_H" in df.columns,
        "sot_A": "sot_A" in df.columns,
    }

    return {"required": required_cols, "optional": optional_cols}
