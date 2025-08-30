from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import joblib
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path(__file__).with_name("random_forest_model.joblib")
DEFAULT_OVER25_MODEL_PATH = Path(__file__).with_name("random_forest_over25_model.joblib")

# Prefer imblearn.Pipeline, fallback na sklearn.Pipeline
try:  # pragma: no cover - optional dependency
    from imblearn.pipeline import Pipeline as _BasePipeline  # type: ignore
except Exception:  # pragma: no cover
    from sklearn.pipeline import Pipeline as _BasePipeline  # type: ignore


class SampleWeightPipeline(_BasePipeline):
    """Pipeline, která propouští sample_weight do posledního estimatoru."""

    def fit(self, X, y=None, sample_weight=None, **fit_params):  # type: ignore[override]
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


# ---------------------------------------------------------------------
# Načtení dat a příprava featur
# ---------------------------------------------------------------------
def _load_matches(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = []
    for csv in sorted(data_dir.glob("*_combined_full_updated.csv")):
        df = pd.read_csv(csv)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No training files found in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("Date", inplace=True)
    return df


def _compute_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    results = {"H": 1, "D": 0, "A": -1}
    df = df.copy()
    df["home_result"] = df["FTR"].map(results)
    df["away_result"] = -df["home_result"]

    df["home_recent_form"] = (
        df.groupby("HomeTeam")["home_result"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )
    df["away_recent_form"] = (
        df.groupby("AwayTeam")["away_result"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )
    return df.drop(columns=["home_result", "away_result"])


def _compute_expected_goals(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    ratings = {team: 1500 for team in teams}
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


def _clip_features(df: pd.DataFrame) -> pd.DataFrame:
    bounds = {
        "home_recent_form": (-1, 1),
        "away_recent_form": (-1, 1),
        "elo_diff": (-400, 400),
        "xg_diff": (-5, 5),
        "home_conceded": (0, 5),
        "away_conceded": (0, 5),
        "conceded_diff": (-5, 5),
        "shots_diff": (-20, 20),
        "shot_target_diff": (-10, 10),
        "corners_diff": (-10, 10),
        "days_since_last_match": (-30, 30),
        "attack_strength_diff": (-2, 2),
        "defense_strength_diff": (-2, 2),
        "home_advantage": (1.0, 1.0),
    }
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Iterable[str], Any]:
    df = _compute_recent_form(df)
    df = _compute_expected_goals(df)
    df = _compute_elo_difference(df)

    # rolling střely/na branku/rohy
    df["home_shots_avg"] = df.groupby("HomeTeam")["HS"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_shots_avg"] = df.groupby("AwayTeam")["AS"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["shots_diff"] = df["home_shots_avg"] - df["away_shots_avg"]

    df["home_shot_target_avg"] = df.groupby("HomeTeam")["HST"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_shot_target_avg"] = df.groupby("AwayTeam")["AST"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["shot_target_diff"] = df["home_shot_target_avg"] - df["away_shot_target_avg"]

    df["home_corners_avg"] = df.groupby("HomeTeam")["HC"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["away_corners_avg"] = df.groupby("AwayTeam")["AC"].transform(
        lambda x: x.shift().rolling(5, min_periods=1).mean()
    )
    df["corners_diff"] = df["home_corners_avg"] - df["away_corners_avg"]

    # odpočinek (rozdíl dní)
    df["home_last"] = df.groupby("HomeTeam")["Date"].shift()
    df["away_last"] = df.groupby("AwayTeam")["Date"].shift()
    df["home_rest"] = (df["Date"] - df["home_last"]).dt.days
    df["away_rest"] = (df["Date"] - df["away_last"]).dt.days
    df["days_since_last_match"] = df["home_rest"] - df["away_rest"]
    df.drop(columns=["home_last", "away_last", "home_rest", "away_rest"], inplace=True)

    # týmové síly z Poisson utilit
    from utils.poisson_utils.stats import calculate_team_strengths
    attack_strength, defense_strength, _ = calculate_team_strengths(df)
    df["attack_strength_diff"] = df["HomeTeam"].map(attack_strength) - df["AwayTeam"].map(attack_strength)
    df["defense_strength_diff"] = df["HomeTeam"].map(defense_strength) - df["AwayTeam"].map(defense_strength)

    # konstantní home advantage (1.0) – je součástí tréninku
    df["home_advantage"] = 1.0

    features = [
        "home_recent_form",
        "away_recent_form",
        "elo_diff",
        "xg_diff",
        "home_conceded",
        "away_conceded",
        "conceded_diff",
        "shots_diff",
        "shot_target_diff",
        "corners_diff",
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

    from sklearn.preprocessing import LabelEncoder
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)
    return X, y, features, label_enc


# ---------------------------------------------------------------------
# Trénink modelů
# ---------------------------------------------------------------------
def train_model(
    data_dir: str | Path = "data",
    n_splits: int = 5,
    recent_years: int | None = None,
    n_iter: int = 20,
    max_samples: int | None = None,
    decay_factor: float | None = None,
    param_distributions: Mapping[str, Iterable[Any]] | None = None,
    balance_classes: bool = False,
) -> Tuple[Any, Iterable[str], Any, float, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train RandomForest (H/D/A) s časovým CV a kalibrací.

    Vrací: (calibrated_model, feature_names, label_encoder, best_cv_score, best_params, per_class_metrics)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.base import clone

    df = _load_matches(data_dir)
    if recent_years is not None and "Date" in df.columns:
        cutoff = df["Date"].max() - pd.DateOffset(years=recent_years)
        df = df[df["Date"] >= cutoff]
    if max_samples is not None:
        df = df.tail(max_samples)

    X, y, feature_names, label_enc = _prepare_features(df)

    # váhy vzorků podle stáří
    sample_weights = None
    if decay_factor is not None and "Date" in df.columns:
        max_date = df["Date"].max()
        age = (max_date - df.loc[X.index, "Date"]).dt.days.to_numpy()
        sample_weights = np.exp(-decay_factor * age)

    # class_weight (volitelné)
    class_weight = None
    if balance_classes:
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {cls: w for cls, w in zip(classes, weights)}

    pipeline = SampleWeightPipeline(
        [("model", RandomForestClassifier(class_weight=class_weight, random_state=42))]
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if param_distributions is None:
        param_distributions = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__bootstrap": [True, False],
        }

    # Optimalizujeme log-loss, aby pravděpodobnosti lépe odpovídaly realitě
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        random_state=42,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    fit_params = {"sample_weight": sample_weights} if sample_weights is not None else {}
    search.fit(X, y, **fit_params)

    best_pipeline = search.best_estimator_
    # RandomizedSearchCV s "neg_log_loss" vrací záporné hodnoty, otočíme znamenko
    score = float(-search.best_score_)
    best_params = search.best_params_

    calibrated_model = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=tscv)
    if sample_weights is not None:
        calibrated_model.fit(X, y, sample_weight=sample_weights)
    else:
        calibrated_model.fit(X, y)

    # per-class precision/recall + Brier score z rollingu přes TimeSeriesSplit
    y_pred = np.empty_like(y)
    y_proba = np.zeros((len(y), len(np.unique(y))))
    for tr, te in tscv.split(X, y):
        mdl = clone(best_pipeline)
        if sample_weights is not None:
            mdl.fit(X.iloc[tr], y[tr], sample_weight=sample_weights[tr])
        else:
            mdl.fit(X.iloc[tr], y[tr])
        y_pred[te] = mdl.predict(X.iloc[te])
        y_proba[te] = mdl.predict_proba(X.iloc[te])

    precisions, recalls, _, _ = precision_recall_fscore_support(
        y, y_pred, labels=np.unique(y)
    )
    from sklearn.metrics import brier_score_loss

    metrics = {}
    for idx, (label, p, r) in enumerate(zip(label_enc.classes_, precisions, recalls)):
        true = (y == idx).astype(int)
        prob = y_proba[:, idx]
        metrics[label] = {
            "precision": float(p),
            "recall": float(r),
            "brier": float(brier_score_loss(true, prob)),
        }

    return calibrated_model, feature_names, label_enc, score, best_params, metrics


def train_over25_model(
    data_dir: str | Path = "data",
    n_splits: int = 5,
    recent_years: int | None = None,
    n_iter: int = 20,
    max_samples: int | None = None,
    param_distributions: Mapping[str, Iterable[Any]] | None = None,
) -> Tuple[Any, Iterable[str], Any, float, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Binary RF pro Over 2.5 (s kalibrací a vyvážením tříd)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import LabelEncoder
    from sklearn.base import clone

    df = _load_matches(data_dir)
    if recent_years is not None and "Date" in df.columns:
        cutoff = df["Date"].max() - pd.DateOffset(years=recent_years)
        df = df[df["Date"] >= cutoff]
    if max_samples is not None:
        df = df.tail(max_samples)

    df["over25"] = np.where(df["FTHG"] + df["FTAG"] > 2.5, "Over 2.5", "Under 2.5")
    X, _, feature_names, _ = _prepare_features(df)
    y_raw = df.loc[X.index, "over25"]

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y_raw)

    # balanced varianta s fallbackem
    try:  # pragma: no cover
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
        from sklearn.pipeline import Pipeline
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weight = {cls: w for cls, w in zip(classes, weights)}
        pipeline = Pipeline(
            [("model", RandomForestClassifier(class_weight=class_weight, random_state=42))]
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if param_distributions is None:
        param_distributions = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__bootstrap": [True, False],
        }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        random_state=42,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    search.fit(X, y)

    best_pipeline = search.best_estimator_
    score = float(-search.best_score_)
    best_params = search.best_params_

    calibrated_model = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=tscv)
    calibrated_model.fit(X, y)

    y_pred = np.empty_like(y)
    y_proba = np.zeros((len(y), len(np.unique(y))))
    for tr, te in tscv.split(X, y):
        mdl = clone(best_pipeline)
        mdl.fit(X.iloc[tr], y[tr])
        y_pred[te] = mdl.predict(X.iloc[te])
        y_proba[te] = mdl.predict_proba(X.iloc[te])

    precisions, recalls, _, _ = precision_recall_fscore_support(
        y, y_pred, labels=np.unique(y)
    )
    from sklearn.metrics import brier_score_loss

    metrics = {}
    for idx, (label, p, r) in enumerate(zip(label_enc.classes_, precisions, recalls)):
        true = (y == idx).astype(int)
        prob = y_proba[:, idx]
        metrics[label] = {
            "precision": float(p),
            "recall": float(r),
            "brier": float(brier_score_loss(true, prob)),
        }

    return calibrated_model, feature_names, label_enc, score, best_params, metrics


# ---------------------------------------------------------------------
# Uložení/načtení, inference
# ---------------------------------------------------------------------
def save_model(
    model: Any,
    feature_names: Iterable[str],
    label_encoder: Any,
    path: str | Path = DEFAULT_MODEL_PATH,
    best_params: Mapping[str, Any] | None = None,
) -> None:
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
    try:
        data = joblib.load(Path(path))
        model = data["model"]
        logger.info("Loaded model type: %s", type(model).__name__)
        return model, data["feature_names"], data["label_encoder"], data.get("best_params", {})
    except Exception:
        logger.warning("Training RandomForest model because %s is missing", path)
        model, feature_names, label_enc, _, params, _ = train_model(
            n_splits=2, n_iter=1, max_samples=200
        )
        save_model(model, feature_names, label_enc, path, params)
        return model, feature_names, label_enc, params


def predict_outcome(
    features: Dict[str, float],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    alpha: float = 0.15,
) -> str:
    """Vrať predikovaný výsledek ('H'/'D'/'A') s tlumením pravděpodobností."""
    probs = predict_proba(features, model_path=model_path, alpha=alpha)
    reverse = {"Home Win": "H", "Draw": "D", "Away Win": "A"}
    return reverse[max(probs, key=probs.get)]


def construct_features_for_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    elo_dict: Mapping[str, float],
) -> Dict[str, float]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values("Date", inplace=True)

    def recent_form(team: str) -> float:
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        vals: list[float] = []
        for _, r in tm.iterrows():
            if r["HomeTeam"] == team:
                vals.append(1 if r["FTR"] == "H" else 0 if r["FTR"] == "D" else -1)
            else:
                vals.append(1 if r["FTR"] == "A" else 0 if r["FTR"] == "D" else -1)
        return float(np.mean(vals)) if vals else 0.0

    def recent_goals(team: str) -> float:
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        goals: list[float] = []
        for _, r in tm.iterrows():
            goals.append(r.get("FTHG", 0) if r["HomeTeam"] == team else r.get("FTAG", 0))
        return float(np.mean(goals)) if goals else 0.0

    def recent_conceded(team: str) -> float:
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
        goals: list[float] = []
        for _, r in tm.iterrows():
            goals.append(r.get("FTAG", 0) if r["HomeTeam"] == team else r.get("FTHG", 0))
        return float(np.mean(goals)) if goals else 0.0

    def last_match_date(team: str) -> pd.Timestamp | None:
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        return None if tm.empty else tm["Date"].max()

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

    h_last = last_match_date(home_team)
    a_last = last_match_date(away_team)
    days_since_last_match = float((h_last - a_last).days) if h_last and a_last else 0.0

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
    model_data: Tuple[Any, Iterable[str], Any]
    | Tuple[Any, Iterable[str], Any, Mapping[str, Any]]
    | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    alpha: float = 0.15,
) -> Dict[str, float]:
    """Vrať pravděpodobnosti {'Home Win','Draw','Away Win'} v procentech (0–100)."""
    if model_data is None:
        model_data = load_model(model_path)
    model, feature_names, label_enc = model_data[:3]
    X = _clip_features(pd.DataFrame([features], columns=feature_names))
    probs = model.predict_proba(X)[0]
    # tlumení k uniformnímu prioru (snižuje extrémy)
    probs = (1 - alpha) * probs + alpha * (1.0 / len(probs))
    probs = probs / probs.sum()
    labels = label_enc.inverse_transform(np.arange(len(probs)))
    mapping = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    return {mapping[lbl]: float(p * 100) for lbl, p in zip(labels, probs)}


def load_over25_model(path: str | Path = DEFAULT_OVER25_MODEL_PATH):
    try:
        data = joblib.load(Path(path))
        return data["model"], data["feature_names"], data.get("label_encoder")
    except Exception:
        logger.warning("Training over/under model because %s is missing", path)
        model, feature_names, label_enc, _, params, _ = train_over25_model(
            n_splits=2, n_iter=1, max_samples=200
        )
        joblib.dump(
            {
                "model": model,
                "feature_names": list(feature_names),
                "label_encoder": label_enc,
                "best_params": params,
            },
            Path(path),
        )
        return model, feature_names, label_enc


def predict_over25_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any] | None = None,
    model_path: str | Path = DEFAULT_OVER25_MODEL_PATH,
    alpha: float = 0.15,
) -> float:
    """Vrať pravděpodobnost (0–100), že padne > 2.5 gólu (Over 2.5)."""
    if model_data is None:
        model_data = load_over25_model(model_path)
    model, feature_names, label_enc = model_data
    X = _clip_features(pd.DataFrame([features], columns=feature_names))
    raw_proba = model.predict_proba(X)[0]
    raw_proba = np.clip(raw_proba, 0.01, 0.99)

    if label_enc is not None:
        expected = np.arange(len(label_enc.classes_))
        model_classes = getattr(model, "classes_", expected)
        probs_full = np.zeros(len(expected))
        for p, cls in zip(raw_proba, model_classes):
            probs_full[int(cls)] = p
        classes = label_enc.inverse_transform(np.arange(len(expected)))
        over_idx = list(classes).index("Over 2.5")
        prob = probs_full[over_idx]
    else:
        # binární model => index 1 je „Over“
        prob = raw_proba[1]

    prob = (1 - alpha) * prob + alpha * 0.5
    return float(prob * 100)


__all__ = [
    "train_model",
    "train_over25_model",
    "save_model",
    "predict_outcome",
    "construct_features_for_match",
    "predict_proba",
    "load_model",
    "load_over25_model",
    "predict_over25_proba",
]
