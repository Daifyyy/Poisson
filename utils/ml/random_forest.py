"""Lightweight Random Forest helpers used by the Streamlit app.

The original project contained a training script in this module which executed
heavy data downloads at import time.  For the purposes of the tests and the
example application we provide small wrappers around pre-trained models stored
on disk.  If those models are missing we fall back to a very small ``DummyModel``
that mimics the scikit-learn API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd

from .dummy_model import DummyModel, SimpleLabelEncoder

DEFAULT_MODEL_PATH = Path(__file__).with_name("random_forest_model.joblib")
DEFAULT_OVER25_MODEL_PATH = Path(__file__).with_name(
    "random_forest_over25_model.joblib"
)


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Tuple[Any, Iterable[str], Any]:
    """Load the outcome model from disk.

    If loading fails, a very small dummy model with a fixed feature set is
    returned so the application continues to work in a limited fashion.
    """

    try:
        data = joblib.load(Path(path))
        return data["model"], data["feature_names"], data.get("label_encoder")
    except Exception:  # pragma: no cover - fallback path
        return DummyModel(), [
            "home_recent_form",
            "away_recent_form",
            "elo_diff",
            "xg_diff",
            "xga_diff",
            "home_conceded",
            "away_conceded",
            "conceded_diff",
            "shots_diff",
            "shot_target_diff",
            "poss_diff",
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
        return data["model"], data["feature_names"], data.get("label_encoder")
    except Exception:  # pragma: no cover - fallback path
        return DummyModel(), [
            "home_recent_form",
            "away_recent_form",
            "elo_diff",
            "xg_diff",
            "xga_diff",
            "home_conceded",
            "away_conceded",
            "conceded_diff",
            "shots_diff",
            "shot_target_diff",
            "poss_diff",
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
) -> Dict[str, float]:
    """Construct feature mapping for a single match from FBR stats.

    The function operates on a DataFrame in the shape produced by
    :mod:`fbrapi_dataset` (``team_H``, ``team_A`` etc.) and calculates
    lightweight rolling metrics for the two sides.  Only a small subset of the
    full training pipeline is implemented so the app can work with minimal
    historical data.
    """

    df = df.copy()
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df.sort_values(date_col, inplace=True)

    def recent_form(team: str) -> float:
        tm = df[(df.get("team_H") == team) | (df.get("team_A") == team)].tail(5)
        vals: list[float] = []
        for _, r in tm.iterrows():
            if r.get("team_H") == team:
                gf, ga = r.get("gf_H", 0), r.get("ga_H", 0)
            else:
                gf, ga = r.get("gf_A", 0), r.get("ga_A", 0)
            vals.append(1 if gf > ga else 0 if gf == ga else -1)
        return float(np.mean(vals)) if vals else 0.0

    def rolling_avg(team: str, col_home: str, col_away: str) -> float:
        tm = df[(df.get("team_H") == team) | (df.get("team_A") == team)].tail(5)
        vals: list[float] = []
        for _, r in tm.iterrows():
            vals.append(r.get(col_home, 0.0) if r.get("team_H") == team else r.get(col_away, 0.0))
        return float(np.mean(vals)) if vals else 0.0

    def last_match_date(team: str) -> pd.Timestamp | None:
        tm = df[(df.get("team_H") == team) | (df.get("team_A") == team)]
        return None if tm.empty else tm[date_col].max()

    home_form = recent_form(home_team)
    away_form = recent_form(away_team)

    home_xg = rolling_avg(home_team, "xg_H", "xg_A")
    away_xg = rolling_avg(away_team, "xg_H", "xg_A")
    xg_diff = home_xg - away_xg

    home_xga = rolling_avg(home_team, "xga_H", "xga_A")
    away_xga = rolling_avg(away_team, "xga_H", "xga_A")
    xga_diff = home_xga - away_xga

    home_conc = rolling_avg(home_team, "ga_H", "ga_A")
    away_conc = rolling_avg(away_team, "ga_H", "ga_A")
    conceded_diff = home_conc - away_conc

    shots_diff = rolling_avg(home_team, "shots_H", "shots_A") - rolling_avg(away_team, "shots_H", "shots_A")
    shot_target_diff = rolling_avg(home_team, "sot_H", "sot_A") - rolling_avg(away_team, "sot_H", "sot_A")
    poss_diff = rolling_avg(home_team, "poss_H", "poss_A") - rolling_avg(away_team, "poss_H", "poss_A")

    h_last = last_match_date(home_team)
    a_last = last_match_date(away_team)
    days_since_last = float((h_last - a_last).days) if h_last and a_last else 0.0

    attack_strength_diff = 0.0
    defense_strength_diff = 0.0
    try:  # pragma: no cover - optional dependency
        from utils.poisson_utils.stats import calculate_team_strengths

        tmp = df[[date_col, "team_H", "team_A", "gf_H", "gf_A"]].rename(
            columns={
                date_col: "Date",
                "team_H": "HomeTeam",
                "team_A": "AwayTeam",
                "gf_H": "FTHG",
                "gf_A": "FTAG",
            }
        )
        atk, dfn, _ = calculate_team_strengths(tmp)
        attack_strength_diff = atk.get(home_team, 0.0) - atk.get(away_team, 0.0)
        defense_strength_diff = dfn.get(home_team, 0.0) - dfn.get(away_team, 0.0)
    except Exception:
        pass

    features = {
        "home_recent_form": home_form,
        "away_recent_form": away_form,
        "elo_diff": float(elo_dict.get(home_team, 1500) - elo_dict.get(away_team, 1500)),
        "xg_diff": xg_diff,
        "xga_diff": xga_diff,
        "home_conceded": home_conc,
        "away_conceded": away_conc,
        "conceded_diff": conceded_diff,
        "shots_diff": shots_diff,
        "shot_target_diff": shot_target_diff,
        "poss_diff": poss_diff,
        # ``corners_diff`` retained for compatibility; possession data is a better
        # proxy in the FBR dataset so corners are not available.
        "corners_diff": 0.0,
        "home_advantage": 1.0,
        "days_since_last_match": days_since_last,
        "attack_strength_diff": attack_strength_diff,
        "defense_strength_diff": defense_strength_diff,
    }
    return features


def predict_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    alpha: float = 0.15,
) -> Dict[str, float]:
    """Return outcome probabilities (Home/Draw/Away) in percent."""

    if model_data is None:
        model_data = load_model(model_path)
    model, feature_names, label_enc = model_data
    X = pd.DataFrame([features], columns=feature_names).fillna(0.0)
    probs = model.predict_proba(X.to_numpy())[0]
    probs = (1 - alpha) * probs + alpha * (1.0 / len(probs))
    labels = label_enc.inverse_transform(np.arange(len(probs)))
    mapping = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    return {mapping.get(lbl, lbl): float(p * 100) for lbl, p in zip(labels, probs)}


def predict_over25_proba(
    features: Dict[str, float],
    model_data: Tuple[Any, Iterable[str], Any] | None = None,
    model_path: str | Path = DEFAULT_OVER25_MODEL_PATH,
    alpha: float = 0.05,
) -> float:
    """Return probability (0–100) that a match finishes Over 2.5 goals."""

    if model_data is None:
        model_data = load_over25_model(model_path)
    model, feature_names, label_enc = model_data
    X = pd.DataFrame([features], columns=feature_names).fillna(0.0)
    raw_proba = model.predict_proba(X.to_numpy())[0]

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
        prob = raw_proba[1]

    prob = (1 - alpha) * prob + alpha * 0.5
    return float(prob * 100)


__all__ = [
    "load_model",
    "load_over25_model",
    "construct_features_for_match",
    "predict_proba",
    "predict_over25_proba",
]
