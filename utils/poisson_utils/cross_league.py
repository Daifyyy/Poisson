import pandas as pd


def calculate_cross_league_team_index(df: pd.DataFrame, league_ratings: pd.DataFrame) -> pd.DataFrame:
    """Return league-adjusted team ratings to compare clubs across leagues.

    Parameters
    ----------
    df: pd.DataFrame
        Team statistics with columns ``league``, ``team``, ``matches`` and
        metrics like ``goals_for``, ``goals_against``, ``xg_for``, ``xg_against``
        (optionally ``shots_for`` and ``shots_against``). Values should be totals
        across the provided matches.
    league_ratings: pd.DataFrame
        Table of league strength ratings with columns ``league`` and ``elo``.

    Returns
    -------
    pd.DataFrame
        Original DataFrame extended with per-match metrics, normalised xG ratio
        and a ``team_index`` scaled by league strength. Higher values indicate a
        stronger team relative to world average.
    """
    metrics = [
        "goals_for",
        "goals_against",
        "xg_for",
        "xg_against",
        "shots_for",
        "shots_against",
    ]

    df = df.copy()
    available = [m for m in metrics if m in df.columns]
    if "matches" not in df.columns or df["matches"].eq(0).any():
        raise ValueError("DataFrame must contain 'matches' column with non-zero values")

    # convert to per-match values (per 90 minutes)
    df[available] = df[available].div(df["matches"], axis=0)

    # xG ratio relative to league average
    if {"xg_for", "xg_against"}.issubset(df.columns):
        df["xg_ratio"] = df["xg_for"] / df["xg_against"].replace(0, pd.NA)
        df["xg_league_avg"] = df.groupby("league")["xg_ratio"].transform("mean")
        df["xg_vs_league"] = df["xg_ratio"] / df["xg_league_avg"]
    else:
        df["xg_vs_league"] = 1.0

    # merge league strength and scale to world average
    df = df.merge(league_ratings, on="league", how="left")
    if df["elo"].isna().any():
        raise ValueError("Missing ELO rating for some leagues")
    elo_mean = league_ratings["elo"].mean()
    df["xg_vs_world"] = df["xg_vs_league"] * (df["elo"] / elo_mean)

    # simple offensive/defensive ratings
    df["off_rating"] = df["goals_for"] / df["goals_against"].replace(0, pd.NA)
    df["def_rating"] = df["xg_against"] / df["xg_for"].replace(0, pd.NA)

    df["team_index"] = df["off_rating"] * (1 / df["def_rating"]) * df["xg_vs_world"]
    return df
