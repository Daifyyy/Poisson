"""Cross-league cup match predictions.

This module provides a lightweight helper for predicting one-off cup matches
between clubs from different leagues. It relies on the cross-league strength
index produced by :func:`utils.poisson_utils.cross_league.calculate_cross_league_team_index`
and uses a simple Poisson model to transform the rating difference into match
outcomes.

Assumptions
-----------
* Baseline scoring rate ``BASE_GOALS`` is the expected goals for an average
  team on neutral ground.
* Home advantage is modelled as a fixed boost ``HOME_ADVANTAGE`` to the home
  team's ``team_index``. Set this constant to ``0`` for neutral venues.
* ``INDEX_TO_XG`` scales the rating difference to an expected-goal
  differential. The constants are calibrated heuristically and can be tweaked
  for different datasets.
* Goal counts are assumed to follow a Poisson distribution and probabilities
  are calculated from a score matrix truncated at ``MAX_GOALS`` goals per
  team.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .prediction import poisson_prediction

# Constants documented above. They can be modified by downstream code if a
# different calibration is desired.
HOME_ADVANTAGE = 0.15  # rating boost for the home side
BASE_GOALS = 1.35  # world average goals per team
INDEX_TO_XG = 0.4  # rating to expected goals multiplier
MAX_GOALS = 6  # truncation for the Poisson matrix


def _lookup_team(team: str, df: pd.DataFrame) -> pd.Series:
    """Return row for ``team`` from ``df`` raising ``KeyError`` if missing."""

    match = df.loc[df["team"] == team]
    if match.empty:
        raise KeyError(f"Team '{team}' not found in cross_league_df")
    return match.iloc[0]


def predict_cup_match(home_team: str, away_team: str, cross_league_df: pd.DataFrame) -> Dict[str, float]:
    """Predict an inter-league cup match using team strength indices.

    Parameters
    ----------
    home_team, away_team : str
        Club names to match against ``cross_league_df``'s ``team`` column.
    cross_league_df : pandas.DataFrame
        DataFrame containing at least ``team`` and ``team_index`` columns and
        optionally ``xg_vs_world``. These are typically produced by
        :func:`calculate_cross_league_team_index`.

    Returns
    -------
    dict
        Dictionary with expected goals and win/draw percentages for the home
        and away side. Keys:

        ``home_exp_goals``
            Expected goals for the home team.
        ``away_exp_goals``
            Expected goals for the away team.
        ``home_win_pct``
            Probability of the home team winning (percent).
        ``draw_pct``
            Probability of a draw (percent).
        ``away_win_pct``
            Probability of the away team winning (percent).

    Notes
    -----
    * Home advantage is incorporated through the ``HOME_ADVANTAGE`` constant.
      For neutral venues, set this constant to ``0`` before calling the
      function.
    * ``INDEX_TO_XG`` controls the relationship between rating difference and
      expected-goal difference. Increasing it leads to more extreme scorelines.
    * Probabilities are derived from a Poisson score matrix truncated at
      ``MAX_GOALS`` goals per team. The truncation introduces negligible error
      for typical football scores.
    """

    home_row = _lookup_team(home_team, cross_league_df)
    away_row = _lookup_team(away_team, cross_league_df)

    home_index = float(home_row["team_index"])
    away_index = float(away_row["team_index"])
    diff = home_index - away_index + HOME_ADVANTAGE

    # Expected goals based on rating difference.
    goal_diff = diff * INDEX_TO_XG
    home_exp = BASE_GOALS + goal_diff
    away_exp = BASE_GOALS - goal_diff

    # Optional adjustment using xG differential vs world average.
    if "xg_vs_world" in home_row and "xg_vs_world" in away_row:
        if not pd.isna(home_row["xg_vs_world"]) and not pd.isna(away_row["xg_vs_world"]):
            adjust = (float(home_row["xg_vs_world"]) - float(away_row["xg_vs_world"])) / 2
            home_exp += adjust
            away_exp -= adjust

    # Prevent negative expected goals which would invalidate Poisson model.
    home_exp = max(home_exp, 0.01)
    away_exp = max(away_exp, 0.01)

    matrix = poisson_prediction(home_exp, away_exp, max_goals=MAX_GOALS)
    # Lower triangle (excluding diagonal) represents home team wins when rows
    # correspond to home goals and columns to away goals.
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    # Upper triangle (excluding diagonal) captures away team wins.
    away_win = float(np.triu(matrix, 1).sum())

    return {
        "home_exp_goals": round(home_exp, 2),
        "away_exp_goals": round(away_exp, 2),
        "home_win_pct": round(home_win * 100, 2),
        "draw_pct": round(draw * 100, 2),
        "away_win_pct": round(away_win * 100, 2),
    }
