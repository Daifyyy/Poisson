import pandas as pd
import warnings

from .elo import calculate_elo_ratings
from .team_analysis import calculate_strength_of_schedule


# Historically calibrated world average ELO rating. Using a fixed reference
# ensures that ``team_index`` values remain comparable across seasons and
# different sets of leagues.
WORLD_ELO_MEAN = 1500


def calculate_cross_league_team_index(
    df: pd.DataFrame,
    league_ratings: pd.DataFrame,
    matches: pd.DataFrame,
    min_matches: int = 10,
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
        differential, offensive/defensive ratings and a ``team_index`` scaled by
        league strength and opponent quality. Higher values indicate a stronger
        team relative to world average.
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
    per_match = df[available].div(df["matches"], axis=0)

    # league average for each metric (weighted by match count)
    league_totals = df.groupby("league")[available + ["matches"]].sum()
    league_avgs = league_totals[available].div(league_totals["matches"], axis=0)
    league_avg_df = (
        df[["league"]]
        .merge(league_avgs, left_on="league", right_index=True, how="left")
        [available]
    )

    # blend team metrics with league average when sample size is small
    weight = (df["matches"].clip(upper=min_matches) / min_matches)
    df[available] = per_match.mul(weight, axis=0) + league_avg_df.mul(1 - weight, axis=0)

    # xG differential normalised by league mean/std
    has_xg = {"xg_for", "xg_against"}.issubset(df.columns) and not (
        df.get("xg_for").isna().any() or df.get("xg_against").isna().any()
    )
    if has_xg:
        df["xg_diff"] = df["xg_for"] - df["xg_against"]
        grp = df.groupby("league")["xg_diff"]
        df["xg_diff_norm"] = (
            df["xg_diff"] - grp.transform("mean")
        ) / grp.transform("std").replace(0, pd.NA)
    elif {"goals_for", "goals_against"}.issubset(df.columns):
        warnings.warn(
            "Missing xG data; falling back to goal differential which may be less accurate",
            UserWarning,
        )
        df["goal_diff"] = df["goals_for"] - df["goals_against"]
        grp = df.groupby("league")["goal_diff"]
        df["xg_diff_norm"] = (
            df["goal_diff"] - grp.transform("mean")
        ) / grp.transform("std").replace(0, pd.NA)
    else:
        raise ValueError(
            "calculate_cross_league_team_index requires 'xg_for'/'xg_against' or 'goals_for'/'goals_against' columns"
        )

    # merge league strength and scale to world average
    missing_leagues = set(df["league"].unique()) - set(league_ratings["league"])
    if missing_leagues:
        missing = ", ".join(sorted(missing_leagues))
        raise ValueError(f"Missing league ratings for: {missing}")
    df = df.merge(league_ratings, on="league", how="left")
    if df["elo"].isna().any():
        raise ValueError("Missing ELO rating for some leagues")
    league_factor = df["elo"] / WORLD_ELO_MEAN

    # offensive and defensive ratings vs league norms (higher is better)
    off_components = []
    def_components = []
    if "goals_for" in available:
        off_components.append(df["goals_for"] - league_avg_df["goals_for"])
    if "xg_for" in available:
        off_components.append(df["xg_for"] - league_avg_df["xg_for"])
    if "goals_against" in available:
        def_components.append(league_avg_df["goals_against"] - df["goals_against"])
    if "xg_against" in available:
        def_components.append(league_avg_df["xg_against"] - df["xg_against"])

    df["off_rating"] = sum(off_components) / len(off_components) if off_components else 0.0
    df["def_rating"] = sum(def_components) / len(def_components) if def_components else 0.0

    # expected goals differential vs world average opponent
    if has_xg:
        df["xg_vs_world"] = (df["xg_for"] - df["xg_against"]) * league_factor
    elif {"goals_for", "goals_against"}.issubset(df.columns):
        df["xg_vs_world"] = (df["goals_for"] - df["goals_against"]) * league_factor
    else:
        # has_xg already validated presence of goal columns earlier; this branch
        # is defensive should the function be modified in future
        raise ValueError(
            "Missing data to compute team differential: provide xG or goal metrics"
        )

    df["team_index"] = (
        0.5 * df["xg_diff_norm"] + 0.5 * df["team_elo_rel"] + 0.1 * df["sos"]
    ) * league_factor
    return df
