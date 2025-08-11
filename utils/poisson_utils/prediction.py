import logging
import math
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .data import prepare_df

logger = logging.getLogger(__name__)


def poisson_prediction(home_exp_goals: float, away_exp_goals: float, max_goals: int = 6) -> np.ndarray:
    """Vrací Poissonovu pravděpodobnost výsledků do maximálního počtu gólů."""
    home_goals_probs = [poisson_pmf(home_exp_goals, i) for i in range(max_goals + 1)]
    away_goals_probs = [poisson_pmf(away_exp_goals, i) for i in range(max_goals + 1)]
    matrix = np.outer(home_goals_probs, away_goals_probs)
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


def poisson_over25_probability(home_exp, away_exp):
    matrix = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            matrix[i][j] = poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)
    prob_over = sum(matrix[i][j] for i in range(7) for j in range(7) if i + j > 2.5)
    return round(prob_over * 100, 2)


def expected_goals_vs_similar_elo_weighted(df, home_team, away_team, elo_dict, elo_tolerance=50):
    df = prepare_df(df)

    elo_home = elo_dict.get(home_team, 1500)
    elo_away = elo_dict.get(away_team, 1500)

    today = df['Date'].max()
    df = df.assign(
        HomeELO=df['HomeTeam'].map(elo_dict),
        AwayELO=df['AwayTeam'].map(elo_dict),
        days_ago=lambda x: (today - x['Date']).dt.days,
        weight=lambda x: 1 / (x['days_ago'] + 1),
    )

    home_mask = df['HomeTeam'] == home_team
    away_mask = df['AwayTeam'] == away_team

    df_home_relevant = df[home_mask & (abs(df['AwayELO'] - elo_away) <= elo_tolerance)]
    df_away_relevant = df[away_mask & (abs(df['HomeELO'] - elo_home) <= elo_tolerance)]

    df_home_all = df[(home_mask | (df['AwayTeam'] == home_team)) &
                     ((abs(df['AwayELO'] - elo_away) <= elo_tolerance) | (abs(df['HomeELO'] - elo_away) <= elo_tolerance))]

    df_away_all = df[((df['HomeTeam'] == away_team) | away_mask) &
                     ((abs(df['HomeELO'] - elo_home) <= elo_tolerance) | (abs(df['AwayELO'] - elo_home) <= elo_tolerance))]

    league_avg_home = df['FTHG'].mean()
    league_avg_away = df['FTAG'].mean()

    def weighted_stat(goals, weights):
        return np.average(goals, weights=weights) if len(goals) > 0 else 1.0

    gf_home = weighted_stat(df_home_relevant['FTHG'], df_home_relevant['weight'])
    ga_home = weighted_stat(df_home_relevant['FTAG'], df_home_relevant['weight'])

    gf_away = weighted_stat(df_away_relevant['FTAG'], df_away_relevant['weight'])
    ga_away = weighted_stat(df_away_relevant['FTHG'], df_away_relevant['weight'])

    gf_home_all = weighted_stat(
        np.where(
            df_home_all['HomeTeam'] == home_team,
            df_home_all['FTHG'],
            df_home_all['FTAG'],
        ),
        df_home_all['weight'],
    )
    ga_home_all = weighted_stat(
        np.where(
            df_home_all['HomeTeam'] == home_team,
            df_home_all['FTAG'],
            df_home_all['FTHG'],
        ),
        df_home_all['weight'],
    )

    gf_away_all = weighted_stat(
        np.where(
            df_away_all['AwayTeam'] == away_team,
            df_away_all['FTAG'],
            df_away_all['FTHG'],
        ),
        df_away_all['weight'],
    )
    ga_away_all = weighted_stat(
        np.where(
            df_away_all['AwayTeam'] == away_team,
            df_away_all['FTHG'],
            df_away_all['FTAG'],
        ),
        df_away_all['weight'],
    )

    def compute_expected(gf, ga_opp, l_home, l_away):
        return l_home * (gf / l_home) * (ga_opp / l_away)

    home_exp_home = compute_expected(gf_home, ga_away, league_avg_home, league_avg_away)
    away_exp_away = compute_expected(gf_away, ga_home, league_avg_away, league_avg_home)

    home_exp_all = compute_expected(gf_home_all, ga_away_all, league_avg_home, league_avg_away)
    away_exp_all = compute_expected(gf_away_all, ga_home_all, league_avg_away, league_avg_home)

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
