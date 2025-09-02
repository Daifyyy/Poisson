"""Lightweight Random Forest helpers used by the Streamlit app.

Updated to better handle different data providers and improved error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional

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
        home_matches = df[df.get("team_H") == team]
        away_matches = df[df.get("team_A") == team]
        all_matches = pd.concat([home_matches, away_matches]).sort_values(date_col)
        recent_matches = all_matches.tail(window)
        
        if debug:
            print(f"{team} recent matches: {len(recent_matches)}")
        
        vals: list[float] = []
        for _, r in recent_matches.iterrows():
            if r.get("team_H") == team:
                gf, ga = r.get("gf_H", 0), r.get("ga_H", 0)
            else:
                gf, ga = r.get("gf_A", 0), r.get("ga_A", 0)
            
            if pd.isna(gf) or pd.isna(ga):
                continue
                
            vals.append(1 if gf > ga else 0 if gf == ga else -1)
        
        return float(np.mean(vals)) if vals else 0.0

    def rolling_avg(team: str, col_home: str, col_away: str, window: int = 5) -> float:
        """Calculate rolling average for a stat."""
        home_matches = df[df.get("team_H") == team]
        away_matches = df[df.get("team_A") == team]
        all_matches = pd.concat([home_matches, away_matches]).sort_values(date_col)
        recent_matches = all_matches.tail(window)
        
        vals: list[float] = []
        for _, r in recent_matches.iterrows():
            if r.get("team_H") == team:
                val = r.get(col_home, 0.0)
            else:
                val = r.get(col_away, 0.0)
            
            if not pd.isna(val):
                vals.append(float(val))
        
        return float(np.mean(vals)) if vals else 0.0

    def last_match_date(team: str) -> Optional[pd.Timestamp]:
        """Get date of last match for a team."""
        home_matches = df[df.get("team_H") == team]
        away_matches = df[df.get("team_A") == team]
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

        home_conc = rolling_avg(home_team, "ga_H", "ga_A")
        away_conc = rolling_avg(away_team, "ga_H", "ga_A")
        conceded_diff = home_conc - away_conc

        shots_diff = rolling_avg(home_team, "shots_H", "shots_A") - rolling_avg(away_team, "shots_H", "shots_A")
        shot_target_diff = rolling_avg(home_team, "sot_H", "sot_A") - rolling_avg(away_team, "sot_H", "sot_A")
        poss_diff = rolling_avg(home_team, "poss_H", "poss_A") - rolling_avg(away_team, "poss_H", "poss_A")

        h_last = last_match_date(home_team)
        a_last = last_match_date(away_team)
        days_since_last = float((h_last - a_last).days) if h_last and a_last else 0.0

        # Try to calculate attack/defense strength
        attack_strength_diff = 0.0
        defense_strength_diff = 0.0
        try:
            from utils.poisson_utils.stats import calculate_team_strengths

            required_cols = [date_col, "team_H", "team_A", "gf_H", "gf_A"]
            if all(col in df.columns for col in required_cols):
                tmp = df[required_cols].rename(
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
        except Exception as e:
            if debug:
                print(f"Could not calculate team strengths: {e}")

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
            "corners_diff": 0.0,  # Not available in FBR dataset
            "home_advantage": 1.0,
            "days_since_last_match": days_since_last,
            "attack_strength_diff": attack_strength_diff,
            "defense_strength_diff": defense_strength_diff,
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
        "home_conceded": 1.0,
        "away_conceded": 1.0,
        "conceded_diff": 0.0,
        "shots_diff": 0.0,
        "shot_target_diff": 0.0,
        "poss_diff": 0.0,
        "corners_diff": 0.0,
        "home_advantage": 1.0,
        "days_since_last_match": 0.0,
        "attack_strength_diff": 0.0,
        "defense_strength_diff": 0.0,
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
        "team_H": "team_H" in df.columns,
        "team_A": "team_A" in df.columns,
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
        "poss_H": "poss_H" in df.columns,
        "poss_A": "poss_A" in df.columns,
    }
    
    return {"required": required_cols, "optional": optional_cols}


def create_sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    
    data = {
        "date": pd.date_range("2024-01-01", periods=10, freq="W"),
        "team_H": ["Arsenal", "Chelsea", "Arsenal", "Liverpool", "Chelsea"] * 2,
        "team_A": ["Chelsea", "Arsenal", "Liverpool", "Arsenal", "Liverpool"] * 2,
        "gf_H": [2, 1, 3, 0, 2, 1, 2, 1, 0, 3],
        "gf_A": [1, 2, 1, 1, 1, 1, 0, 2, 2, 1],
        "ga_H": [1, 2, 1, 1, 1, 1, 0, 2, 2, 1],
        "ga_A": [2, 1, 3, 0, 2, 1, 2, 1, 0, 3],
        "xg_H": [1.8, 0.9, 2.7, 0.3, 1.9, 0.8, 1.7,
