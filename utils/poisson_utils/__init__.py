# from ..poisson_utils import scoreline_variance_warning, combined_form_tempo_warning, conflict_style_warning

from .data import (
    prepare_df,
    load_data,
    detect_current_season,
    get_last_n_matches,
    load_cup_matches,
    load_cup_team_stats,
)

from .stats import (
    calculate_points,
    add_btts_column,
    calculate_team_strengths,
    classify_team_strength,
    aggregate_team_stats,
    compute_form_trend,
    compute_score_stats,
)

from .prediction import (
    poisson_prediction,
    calculate_expected_points,
    prob_to_odds,
    poisson_over25_probability,
    expected_goals_vs_similar_elo_weighted,
)

from .elo import (
    calculate_elo_ratings,
    elo_history,
    calculate_elo_changes,
    detect_risk_factors,
    detect_positive_factors,
    calculate_warning_index,
    detect_overperformance_and_momentum,
)

from .xg import (
    calculate_team_pseudo_xg,
    calculate_pseudo_xg_for_team,
    expected_goals_weighted_by_elo,
    poisson_prediction_matrix,
    over_under_prob,
    bt_btts_prob,
    match_outcomes_prob,
    get_goal_probabilities,
    get_top_scorelines,
    btts_prob,
    expected_team_stats_weighted_by_elo,
    additional_opportunities_prob,
)

from .team_analysis import (
    calculate_form_emojis,
    calculate_conceded_goals,
    calculate_recent_team_form,
    calculate_expected_and_actual_points,
    calculate_strength_of_schedule,
    analyze_opponent_strength,
    get_head_to_head_stats,
    merged_home_away_opponent_form,
    expected_goals_combined_homeaway_allmatches,
    expected_goals_weighted_by_home_away,
    get_team_card_stats,
    generate_team_comparison,
    render_team_comparison_section,
)

from .match_style import (
    calculate_match_tempo,
    calculate_match_style_score_per_match,
    calculate_gii_zscore,
    get_team_average_gii,
    calculate_team_styles,
    intensity_score_to_emoji,
    form_points_to_emoji,
    expected_match_style_score,
    expected_match_tempo,
    tempo_to_emoji,
    get_team_style_vs_opponent_type,
    calculate_attack_volume,
    calculate_attack_efficiency,
    calculate_full_attacking_pressure,
    calculate_advanced_team_metrics,
    calculate_team_extra_stats,
    get_team_record,
    analyze_team_profile,
)

from .corners import (
    expected_corners,
    poisson_corner_matrix,
    corner_over_under_prob,
)

from .xg_sources import get_team_xg_xga
from .cross_league import calculate_cross_league_team_index
from .cup_predictions import predict_cup_match
from .mdi import calculate_mdi
