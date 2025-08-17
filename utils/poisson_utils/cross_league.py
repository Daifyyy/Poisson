import pandas as pd
from .elo import calculate_elo_ratings
from .team_analysis import calculate_strength_of_schedule


def calculate_cross_league_team_index(
    df: pd.DataFrame, league_ratings: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """Return league-adjusted team ratings to compare clubs across leagues.

    Parameters
    ----------
    df : pd.DataFrame
        Team statistics with columns ``league``, ``team``, ``matches`` and
        metrics like ``goals_for``, ``goals_against``, ``xg_for`` and
        ``xg_against`` (optionally ``shots_for`` and ``shots_against``). Values
        should be totals across the provided matches.
    league_ratings : pd.DataFrame
        Table of league strength ratings with columns ``league`` and ``elo``.
    matches : pd.DataFrame
        Match-level results used to compute ELO ratings. Must contain columns
        ``HomeTeam``, ``AwayTeam``, ``FTHG`` and ``FTAG``.

    Returns
    -------
    pd.DataFrame
        Original DataFrame extended with per-match metrics, normalised xG
        differential and a ``team_index`` scaled by league strength and
        opponent quality. Higher values indicate a stronger team relative to
        world average.
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

    # compute team ELO ratings and relative strength within each league
    elo_dict = calculate_elo_ratings(matches)
    elo_df = pd.DataFrame(list(elo_dict.items()), columns=["team", "team_elo"])
    df = df.merge(elo_df, on="team", how="left")
    df["league_elo_mean"] = df.groupby("league")["team_elo"].transform("mean")
    df["team_elo_rel"] = df["team_elo"] / df["league_elo_mean"]

    # strength of schedule (opponent quality) z-score per team
    sos_dict = calculate_strength_of_schedule(matches, metric="elo")
    df["sos"] = df["team"].map(sos_dict).fillna(0)

    # convert to per-match values (per 90 minutes)
    df[available] = df[available].div(df["matches"], axis=0)

    # xG differential normalised by league mean/std
    if {"xg_for", "xg_against"}.issubset(df.columns):
        df["xg_diff"] = df["xg_for"] - df["xg_against"]
        grp = df.groupby("league")["xg_diff"]
        df["xg_diff_norm"] = (df["xg_diff"] - grp.transform("mean")) / grp.transform("std").replace(0, pd.NA)
    else:
        df["xg_diff_norm"] = 0.0

    # merge league strength and scale to world average
    df = df.merge(league_ratings, on="league", how="left")
    if df["elo"].isna().any():
        raise ValueError("Missing ELO rating for some leagues")
    elo_mean = league_ratings["elo"].mean()
    league_factor = df["elo"] / elo_mean

    df["team_index"] = (
        0.5 * df["xg_diff_norm"] + 0.5 * df["team_elo_rel"] + 0.1 * df["sos"]
    ) * league_factor
    return df
