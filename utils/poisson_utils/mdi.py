from __future__ import annotations

from typing import Dict

import pandas as pd


WEIGHTS = {
    "shots": 0.30,
    "shots_on_target": 0.25,
    "corners": 0.15,
    "fouls": 0.10,
    "yellow_cards": 0.10,
    "red_cards": 0.10,
}


def calculate_mdi(
    row: pd.Series, league_avgs: Dict[str, float], opponent_strength: float
) -> float:
    """Vypočítá index dominance zápasu (MDI).

    Parametry
    ---------
    row : pd.Series
        Řádek se statistikami zápasu (HS/AS, HST/AST, HC/AC, HF/AF, HY/AY, HR/AR).
    league_avgs : dict
        Průměrné hodnoty ligy pro uvedené statistiky.
    opponent_strength : float
        Koeficient síly soupeře (slabý < 1, silný > 1).

    Returns
    -------
    float
        Hodnota MDI v rozmezí 0–100.
    """

    relevant_cols = [
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HF",
        "AF",
        "HY",
        "AY",
        "HR",
        "AR",
    ]
    if all(pd.isna(row.get(col)) or row.get(col, 0) == 0 for col in relevant_cols):
        return 50.0

    def _norm(col: str) -> float:
        avg = league_avgs.get(col) or 1
        return row.get(col, 0) / avg

    score = 0.0

    # Statistiky, kde vyšší hodnota je pozitivní
    for home_col, away_col, weight_key in [
        ("HS", "AS", "shots"),
        ("HST", "AST", "shots_on_target"),
        ("HC", "AC", "corners"),
    ]:
        home_norm = _norm(home_col)
        away_norm = _norm(away_col)
        score += (home_norm - away_norm) * WEIGHTS[weight_key]

    # Statistiky, kde nižší hodnota je pozitivní
    for home_col, away_col, weight_key in [
        ("HF", "AF", "fouls"),
        ("HY", "AY", "yellow_cards"),
        ("HR", "AR", "red_cards"),
    ]:
        home_norm = _norm(home_col)
        away_norm = _norm(away_col)
        score += (away_norm - home_norm) * WEIGHTS[weight_key]

    score *= 100 * opponent_strength
    return float(max(0.0, min(100.0, score)))
