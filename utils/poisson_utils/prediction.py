import logging
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .data import prepare_df

logger = logging.getLogger(__name__)


def poisson_prediction(home_exp_goals: float, away_exp_goals: float, max_goals: int = 6) -> np.ndarray:
    """Vrací Poissonovu pravděpodobnost výsledků do maximálního počtu gólů."""
    goals = np.arange(max_goals + 1)
    home_probs = poisson.pmf(goals, home_exp_goals)
    away_probs = poisson.pmf(goals, away_exp_goals)
    matrix = np.outer(home_probs, away_probs)
    return matrix


def poisson_pmf(lmbda: float, k: int) -> float:
    """Poissonova pravděpodobnostní funkce."""
    return (lmbda ** k) * np.exp(-lmbda) / math.factorial(k)


def prob_to_odds(prob: float) -> str:
    """Převede pravděpodobnost (v procentech) na desetinný kurz."""
    if prob <= 0:
        return "-"
    decimal_odds = 100 / prob
    return f"{decimal_odds:.2f}"


def calculate_expected_points(outcomes: dict) -> dict:
    """Calculate expected points based on outcome probabilities."""
    home_xp = (outcomes['Home Win'] / 100) * 3 + (outcomes['Draw'] / 100) * 1
    away_xp = (outcomes['Away Win'] / 100) * 3 + (outcomes['Draw'] / 100) * 1
    return {
        'Home xP': round(home_xp, 1),
        'Away xP': round(away_xp, 1)
    }


def poisson_over25_probability(home_exp: float, away_exp: float, max_goals: int = 6) -> float:
    """Calculate probability of more than 2.5 total goals.

    Args:
        home_exp: Expected home goals.
        away_exp: Expected away goals.
        max_goals: Maximum number of goals to consider per team.

    Returns:
        Probability (in %) that total goals exceed 2.5.
    """
    goals = np.arange(0, max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, home_exp), poisson.pmf(goals, away_exp))
    prob_over = matrix[
        np.add.outer(range(max_goals + 1), range(max_goals + 1)) >= 3
    ].sum()
    return round(prob_over * 100, 2)


def _assign_elo_and_weight(df: pd.DataFrame, elo_dict: Dict[str, float]) -> pd.DataFrame:
    """Add ELO ratings and recency-based weights to the dataframe."""
    today = df["Date"].max()
    return df.assign(
        HomeELO=df["HomeTeam"].map(elo_dict),
        AwayELO=df["AwayTeam"].map(elo_dict),
        days_ago=lambda x: (today - x["Date"]).dt.days,
        weight=lambda x: 1 / (x["days_ago"] + 1),
    )


def _weighted_stat(goals: pd.Series, weights: pd.Series) -> float:
    """Safely compute a weighted average for a series of goals."""
    return float(np.average(goals, weights=weights)) if len(goals) > 0 else 1.0


def _expected_goals(
    goals_for: float,
    goals_against_opponent: float,
    league_avg_for: float,
    league_avg_against: float,
) -> float:
    """Compute expected goals normalized by league averages."""
    return league_avg_for * (goals_for / league_avg_for) * (
        goals_against_opponent / league_avg_against
    )


def expected_goals_vs_similar_elo_weighted(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    elo_dict: Dict[str, float],
    elo_tolerance: int = 50,
) -> Tuple[float, float]:
    """Estimate expected goals against similarly rated opponents.

    Historical results are filtered to matches where the opponent's ELO rating is
    within ``elo_tolerance`` of the upcoming opponent. Older matches are
    down-weighted so that recent games influence the result more heavily. The
    function calculates attacking and defensive strength in both the specific
    home/away role and across all matches against comparable opponents, then
    averages these estimates to produce final expected goals for the home and
    away teams.

    Args:
        df: Match data containing at least ``Date``, ``HomeTeam``, ``AwayTeam``,
            ``FTHG`` and ``FTAG`` columns.
        home_team: Name of the home team.
        away_team: Name of the away team.
        elo_dict: Mapping of teams to their ELO ratings.
        elo_tolerance: Maximum difference in ELO to consider an opponent similar.

    Returns:
        A tuple ``(home_expected, away_expected)`` with the expected goals for
        the home and away team.
    """
    df = prepare_df(df)
    df = _assign_elo_and_weight(df, elo_dict)

    elo_home = elo_dict.get(home_team, 1500)
    elo_away = elo_dict.get(away_team, 1500)

    home_mask = df["HomeTeam"] == home_team
    away_mask = df["AwayTeam"] == away_team

    df_home_relevant = df[home_mask & (abs(df["AwayELO"] - elo_away) <= elo_tolerance)]
    df_away_relevant = df[away_mask & (abs(df["HomeELO"] - elo_home) <= elo_tolerance)]

    df_home_all = df[
        (home_mask | (df["AwayTeam"] == home_team))
        & (
            (abs(df["AwayELO"] - elo_away) <= elo_tolerance)
            | (abs(df["HomeELO"] - elo_away) <= elo_tolerance)
        )
    ]

    df_away_all = df[
        ((df["HomeTeam"] == away_team) | away_mask)
        & (
            (abs(df["HomeELO"] - elo_home) <= elo_tolerance)
            | (abs(df["AwayELO"] - elo_home) <= elo_tolerance)
        )
    ]

    league_avg_home = df["FTHG"].mean()
    league_avg_away = df["FTAG"].mean()

    gf_home = _weighted_stat(df_home_relevant["FTHG"], df_home_relevant["weight"])
    ga_home = _weighted_stat(df_home_relevant["FTAG"], df_home_relevant["weight"])

    gf_away = _weighted_stat(df_away_relevant["FTAG"], df_away_relevant["weight"])
    ga_away = _weighted_stat(df_away_relevant["FTHG"], df_away_relevant["weight"])

    gf_home_all = _weighted_stat(
        np.where(
            df_home_all["HomeTeam"] == home_team,
            df_home_all["FTHG"],
            df_home_all["FTAG"],
        ),
        df_home_all["weight"],
    )
    ga_home_all = _weighted_stat(
        np.where(
            df_home_all["HomeTeam"] == home_team,
            df_home_all["FTAG"],
            df_home_all["FTHG"],
        ),
        df_home_all["weight"],
    )

    gf_away_all = _weighted_stat(
        np.where(
            df_away_all["AwayTeam"] == away_team,
            df_away_all["FTAG"],
            df_away_all["FTHG"],
        ),
        df_away_all["weight"],
    )
    ga_away_all = _weighted_stat(
        np.where(
            df_away_all["AwayTeam"] == away_team,
            df_away_all["FTHG"],
            df_away_all["FTAG"],
        ),
        df_away_all["weight"],
    )

    home_exp_home = _expected_goals(gf_home, ga_away, league_avg_home, league_avg_away)
    away_exp_away = _expected_goals(gf_away, ga_home, league_avg_away, league_avg_home)

    home_exp_all = _expected_goals(gf_home_all, ga_away_all, league_avg_home, league_avg_away)
    away_exp_all = _expected_goals(gf_away_all, ga_home_all, league_avg_away, league_avg_home)

    logger.info("📘 ELO-based: Home/Away only")
    logger.info(
        "  HomeExp: %.1f, AwayExp: %.1f → Over 2.5: %s%%",
        home_exp_home,
        away_exp_away,
        poisson_over25_probability(home_exp_home, away_exp_away),
    )

    logger.info("📘 ELO-based: All relevant matches")
    logger.info(
        "  HomeExp: %.1f, AwayExp: %.1f → Over 2.5: %s%%",
        home_exp_all,
        away_exp_all,
        poisson_over25_probability(home_exp_all, away_exp_all),
    )

    combined_home = round((home_exp_home + home_exp_all) / 2, 1)
    combined_away = round((away_exp_away + away_exp_all) / 2, 1)

    logger.info("🎯 ELO-based kombinace")
    logger.info(
        "  FinalExp: %.1f - %.1f → Over 2.5: %s%%",
        combined_home,
        combined_away,
        poisson_over25_probability(combined_home, combined_away),
    )

    return combined_home, combined_away
