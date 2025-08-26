from .data import (
    prepare_df,
    load_data,
    detect_current_season,
    get_last_n_matches,
    ensure_min_season_matches,
)

from .stats import (
    aggregate_team_stats,
    calculate_points,
    add_btts_column,
    calculate_team_strengths,
    classify_team_strength,
    compute_form_trend,
    compute_score_stats,
)

from .prediction import (
    poisson_prediction,
    poisson_pmf,
    prob_to_odds,
    calculate_expected_points,
    poisson_over25_probability,
    expected_goals_vs_similar_elo_weighted,
)

__all__ = [
    "prepare_df",
    "load_data",
    "detect_current_season",
    "get_last_n_matches",
    "ensure_min_season_matches",
    "aggregate_team_stats",
    "calculate_points",
    "add_btts_column",
    "calculate_team_strengths",
    "classify_team_strength",
    "compute_form_trend",
    "compute_score_stats",
    "poisson_prediction",
    "poisson_pmf",
    "prob_to_odds",
    "calculate_expected_points",
    "poisson_over25_probability",
    "expected_goals_vs_similar_elo_weighted",
]
