import pandas as pd


# Descriptions for metrics returned by ``calculate_cross_league_team_index``.
CROSS_LEAGUE_COLUMN_LEGEND = {
    "goals_for_per_match": "Goals scored per match (per 90 minutes)",
    "goals_against_per_match": "Goals conceded per match",
    "xg_for_per_match": "Expected goals for per match",
    "xg_against_per_match": "Expected goals against per match",
    "shots_for_per_match": "Shots taken per match",
    "shots_against_per_match": "Shots conceded per match",
    "xg_ratio_vs_league_avg": "Team xG ratio relative to the average of its league",
    "xg_ratio_vs_world": "League-strength adjusted xG ratio scaled to world average",
    "offensive_ratio": "Goals scored per match divided by goals conceded per match",
    "defensive_ratio": "Expected goals conceded per match divided by expected goals for",
    "cross_league_index": "Composite strength index for comparing clubs across leagues",
}


def calculate_cross_league_team_index(df: pd.DataFrame, league_ratings: pd.DataFrame) -> pd.DataFrame:
    """Return league-adjusted team ratings to compare clubs across leagues.

    The resulting DataFrame contains columns described in
    ``CROSS_LEAGUE_COLUMN_LEGEND``.

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
        and a ``cross_league_index`` scaled by league strength. Higher values
        indicate a stronger team relative to world average.
    """
    base_metrics = [
        "goals_for",
        "goals_against",
        "xg_for",
        "xg_against",
        "shots_for",
        "shots_against",
    ]

    df = df.copy()
    available = [m for m in base_metrics if m in df.columns]
    if "matches" not in df.columns or df["matches"].eq(0).any():
        raise ValueError("DataFrame must contain 'matches' column with non-zero values")

    # convert to per-match values (per 90 minutes) and rename columns
    df[available] = df[available].div(df["matches"], axis=0)
    rename_map = {
        "goals_for": "goals_for_per_match",
        "goals_against": "goals_against_per_match",
        "xg_for": "xg_for_per_match",
        "xg_against": "xg_against_per_match",
        "shots_for": "shots_for_per_match",
        "shots_against": "shots_against_per_match",
    }
    df.rename(columns={k: rename_map[k] for k in available}, inplace=True)

    # xG ratio relative to league average
    if {"xg_for_per_match", "xg_against_per_match"}.issubset(df.columns):
        df["_xg_ratio"] = df["xg_for_per_match"] / df["xg_against_per_match"].replace(0, pd.NA)
        df["_xg_league_avg"] = df.groupby("league")["_xg_ratio"].transform("mean")
        df["xg_ratio_vs_league_avg"] = df["_xg_ratio"] / df["_xg_league_avg"]
        df.drop(columns=["_xg_ratio", "_xg_league_avg"], inplace=True)
    else:
        df["xg_ratio_vs_league_avg"] = 1.0

    # merge league strength and scale to world average
    df = df.merge(league_ratings, on="league", how="left")
    if df["elo"].isna().any():
        raise ValueError("Missing ELO rating for some leagues")
    elo_mean = league_ratings["elo"].mean()
    df["xg_ratio_vs_world"] = df["xg_ratio_vs_league_avg"] * (df["elo"] / elo_mean)

    # simple offensive/defensive ratios
    df["offensive_ratio"] = (
        df["goals_for_per_match"] / df["goals_against_per_match"].replace(0, pd.NA)
    )
    df["defensive_ratio"] = (
        df["xg_against_per_match"] / df["xg_for_per_match"].replace(0, pd.NA)
    )

    df["cross_league_index"] = (
        df["offensive_ratio"] * (1 / df["defensive_ratio"]) * df["xg_ratio_vs_world"]
    )
    return df
