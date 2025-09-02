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
        ], SimpleLabelEncoder()


def construct_features_for_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    elo_dict: Dict[str, float],
) -> Dict[str, float]:
    """Construct a minimal feature mapping for a single match.

    The implementation here is intentionally lightweight; it provides sane
    defaults and only a couple of simple signals so that the helpers can be
    used without requiring the full training pipeline.
    """

    features = {
        "home_recent_form": 0.0,
        "away_recent_form": 0.0,
        "elo_diff": float(elo_dict.get(home_team, 1500) - elo_dict.get(away_team, 1500)),
        "xg_diff": 0.0,
        "home_conceded": 0.0,
        "away_conceded": 0.0,
        "conceded_diff": 0.0,
        "home_advantage": 1.0,
        "days_since_last_match": 0.0,
        "attack_strength_diff": 0.0,
        "defense_strength_diff": 0.0,
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
