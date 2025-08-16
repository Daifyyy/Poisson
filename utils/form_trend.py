"""Utilities for visualizing team form trends.

This module provides a helper function to compute rolling averages of
match points and a simple pseudo-xG approximation, together with ELO
progression for a given team.  The output is intended for plotting
development of a team's performance over time.
"""

from __future__ import annotations

import pandas as pd

from utils.poisson_utils import elo_history


def get_rolling_form(df: pd.DataFrame, team: str, window: int = 5) -> pd.DataFrame:
    """Return rolling averages of points, pseudo-xG and ELO for ``team``.

    Args:
        df:    DataFrame containing match data.
        team:  Team for which the rolling form should be computed.
        window: Rolling window size, defaulting to 5 matches.

    Returns:
        DataFrame with columns ``Date``, ``rolling_points``,
        ``rolling_xg`` and ``ELO``.
    """

    # Keep only matches involving the selected team and sort chronologically
    team_df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    team_df = team_df.sort_values("Date")

    if team_df.empty:
        return pd.DataFrame(columns=["Date", "rolling_points", "rolling_xg", "ELO"])

    def _points(row: pd.Series) -> int:
        is_home = row["HomeTeam"] == team
        gf = row["FTHG"] if is_home else row["FTAG"]
        ga = row["FTAG"] if is_home else row["FTHG"]
        if gf > ga:
            return 3
        if gf == ga:
            return 1
        return 0

    def _xg(row: pd.Series) -> float:
        if row["HomeTeam"] == team:
            return 0.1 * row.get("HS", 0) + 0.3 * row.get("HST", 0)
        return 0.1 * row.get("AS", 0) + 0.3 * row.get("AST", 0)

    team_df["points"] = team_df.apply(_points, axis=1)
    team_df["xg"] = team_df.apply(_xg, axis=1)

    team_df["rolling_points"] = team_df["points"].rolling(window, min_periods=1).mean()
    team_df["rolling_xg"] = team_df["xg"].rolling(window, min_periods=1).mean()

    elo_df = elo_history(team_df, team)
    team_df = team_df.merge(elo_df, on="Date", how="left")

    return team_df[["Date", "rolling_points", "rolling_xg", "ELO"]]

