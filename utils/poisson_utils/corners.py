import numpy as np
import pandas as pd
from scipy.stats import poisson
from .data import prepare_df


def expected_corners(df: pd.DataFrame, home_team: str, away_team: str) -> tuple:
    """Compute expected corner counts for home and away teams.

    The expectation is a simple average of the team's own corner production and
    the opponent's corners conceded in the corresponding venue (home/away).
    If any component is missing, it is ignored in the average and defaulted to 0.
    """

    required_columns = {"HC", "AC", "HomeTeam", "AwayTeam"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns: {', '.join(sorted(missing_columns))}"
        )

    df = prepare_df(df)

    home_for = df[df['HomeTeam'] == home_team]['HC'].mean()
    home_against = df[df['HomeTeam'] == home_team]['AC'].mean()
    away_for = df[df['AwayTeam'] == away_team]['AC'].mean()
    away_against = df[df['AwayTeam'] == away_team]['HC'].mean()

    home_vals = [v for v in [home_for, away_against] if not pd.isna(v)]
    away_vals = [v for v in [away_for, home_against] if not pd.isna(v)]

    home_exp = np.mean(home_vals) if home_vals else 0.0
    away_exp = np.mean(away_vals) if away_vals else 0.0

    return round(home_exp, 2), round(away_exp, 2)


def poisson_corner_matrix(home_exp: float, away_exp: float, max_corners: int = 20) -> np.ndarray:
    """Return Poisson probability matrix for corner counts up to ``max_corners``."""
    home_probs = [poisson.pmf(i, home_exp) for i in range(max_corners + 1)]
    away_probs = [poisson.pmf(i, away_exp) for i in range(max_corners + 1)]
    return np.outer(home_probs, away_probs)


def corner_over_under_prob(matrix: np.ndarray, threshold: float) -> dict:
    """Compute over/under probabilities for total corners given a matrix."""
    size = matrix.shape[0]
    over = 0.0
    for i in range(size):
        for j in range(size):
            if i + j > threshold:
                over += matrix[i, j]
    over_pct = round(over * 100, 2)
    under_pct = round(100 - over_pct, 2)
    return {f"Over {threshold}": over_pct, f"Under {threshold}": under_pct}
