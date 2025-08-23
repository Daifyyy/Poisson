import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from collections.abc import Mapping
from utils.responsive import responsive_columns
from utils.poisson_utils import (
    expected_goals_weighted_by_elo,
    poisson_prediction,
    match_outcomes_prob,
    over_under_prob,
    btts_prob,
    calculate_expected_points,
    analyze_opponent_strength,
    expected_match_style_score,
    intensity_score_to_emoji,
    expected_match_tempo,
    tempo_to_emoji,
    get_top_scorelines,
    get_goal_probabilities,
    detect_risk_factors,
    detect_positive_factors,
    calculate_warning_index,
    expected_team_stats_weighted_by_elo,
    classify_team_strength,
    merged_home_away_opponent_form,
    get_head_to_head_stats,
    calculate_match_tempo,
    get_team_style_vs_opponent_type,
    expected_goals_combined_homeaway_allmatches,
    expected_goals_weighted_by_home_away,
    expected_corners,
    poisson_corner_matrix,
    corner_over_under_prob,
    detect_current_season,
    get_team_xg_xga,
)
from utils.frontend_utils import display_team_status_table
from utils.poisson_utils.match_style import tempo_tag
from utils.radar_chart import plot_style_radar
from utils.poisson_utils.team_analysis import TEAM_COMPARISON_DESC_MAP
from utils.export_utils import generate_excel_analysis_export
from utils.utils_warnings import (
    scoreline_variance_warning,
    combined_form_tempo_warning,
    conflict_style_warning,
    calculate_recent_form_by_matches,
    get_all_match_warnings,
    get_all_positive_signals,
    detect_overperformance_and_momentum
)
from utils.anomaly_detection import (
    calculate_contrarian_risk_score,
    calculate_upset_risk_score,
    colored_risk_tag,
    calculate_confidence_index
)
from utils import bet_db
from utils.ml.random_forest import (
    load_model,
    construct_features_for_match,
    predict_proba,
)

@st.cache_resource
def get_rf_model():
    return load_model()

RF_MODEL, RF_FEATURE_NAMES, RF_LABEL_ENCODER, _ = get_rf_model()

@st.cache_data
def load_upcoming_xg() -> pd.DataFrame:
    """Load upcoming xG workbook with caching.

    If the workbook or its engine (``openpyxl``) is missing, an empty
    ``DataFrame`` is returned so the app can continue gracefully.
    """
    path = "data/Footballxg.com - (F1X) xG Free Upcoming v3.1.xlsx"
    cols = [
        "Date", "Home Team", "Away Team", "xG Home", "xG Away",
        "Home", "Draw", "Away", ">2.5",
    ]
    try:
        return pd.read_excel(path, header=5, usecols=cols)
    except Exception as exc:  # pragma: no cover - safeguards runtime
        st.warning(f"Could not load xG workbook: {exc}")
        return pd.DataFrame(columns=cols)


def lookup_xg_row(
    df: pd.DataFrame, home_team: str, away_team: str
) -> Optional[pd.Series]:
    """Find matching xG row for given teams (case-insensitive)."""
    if df.empty:
        return None
    mask = (df["Home Team"].str.lower() == home_team.lower()) & \
           (df["Away Team"].str.lower() == away_team.lower())
    match = df.loc[mask]
    if match.empty:
        return None
    return match.iloc[0]



def make_poisson_from_xg(row: pd.Series) -> Dict[str, Dict[str, float]]:
    """Generate Poisson outcome probabilities from xG values."""
    matrix = poisson_prediction(row["xG Home"], row["xG Away"])
    outcomes = match_outcomes_prob(matrix)
    over_under = over_under_prob(matrix, 2.5)
    return {"matrix": matrix, "outcomes": outcomes, "over_under": over_under}

@st.cache_data
def get_cached_match_inputs(df_hash,df, home_team, away_team, elo_dict):
    # from utils.poisson_utils import (
    #     expected_goals_weighted_by_elo, poisson_prediction, match_outcomes_prob,
    #     over_under_prob, btts_prob, calculate_expected_points
    # )

    home_exp, away_exp = expected_goals_combined_homeaway_allmatches(df, home_team, away_team,elo_dict)
    #home_exp, away_exp = expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
    matrix = poisson_prediction(home_exp, away_exp)
    outcomes = match_outcomes_prob(matrix)
    # Compute Over/Under probabilities for multiple goal lines
    over_under = {}
    for thr in (1.5, 2.5, 3.5):
        over_under.update(over_under_prob(matrix, thr))
    btts = btts_prob(matrix)
    xpoints = calculate_expected_points(outcomes)

    return {
        "home_exp": home_exp,
        "away_exp": away_exp,
        "matrix": matrix,
        "outcomes": outcomes,
        "over_under": over_under,
        "btts": btts,
        "xpoints": xpoints
    }
@st.cache_data
def get_cached_tempo(df_hash, df, team, opponent_elo, is_home, elo_dict):
    return calculate_match_tempo(df, team, opponent_elo, is_home, elo_dict)



def compute_match_inputs(
    df: pd.DataFrame,
    season_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    gii_dict: Dict[str, float],
    elo_dict: Dict[str, float],
) -> Dict[str, Any]:
    """Compute and cache core inputs for a single match.

    Returns a dictionary with expected goals, probability matrices and
    auxiliary data used throughout the prediction pipeline.
    """
    df_hash = hash(pd.util.hash_pandas_object(df).sum())
    match_data = get_cached_match_inputs(df_hash, df, home_team, away_team, elo_dict)
    season_start = detect_current_season(season_df, prepared=True)[1]
    season = str(season_start.year)
    ws_home = get_team_xg_xga(home_team, season, season_df)
    ws_away = get_team_xg_xga(away_team, season, season_df)

    xg_home = ws_home.get("xg", float("nan"))
    xga_home = ws_home.get("xga", float("nan"))
    xg_away = ws_away.get("xg", float("nan"))
    xga_away = ws_away.get("xga", float("nan"))

    corner_home, corner_away = expected_corners(df, home_team, away_team)

    gii_home = gii_dict.get(home_team)
    gii_away = gii_dict.get(away_team)
    expected_gii = round(((gii_home or 0) + (gii_away or 0)) / 2, 1)
    expected_tempo = expected_match_tempo(
        df,
        home_team,
        away_team,
        elo_dict,
        match_data["home_exp"],
        match_data["away_exp"],
        xg_home,
        xg_away,
        xga_home,
        xga_away,
    )

    tempo_home = get_cached_tempo(df_hash, df, home_team, elo_dict.get(away_team, 1500), True, elo_dict)
    tempo_away = get_cached_tempo(df_hash, df, away_team, elo_dict.get(home_team, 1500), False, elo_dict)

    return {
        "df_hash": df_hash,
        **match_data,
        "xg_home": xg_home,
        "xga_home": xga_home,
        "xg_away": xg_away,
        "xga_away": xga_away,
        "expected_gii": expected_gii,
        "expected_tempo": expected_tempo,
        "tempo_home": tempo_home,
        "tempo_away": tempo_away,
        "corner_home_exp": corner_home,
        "corner_away_exp": corner_away,
    }


def render_warnings(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    matrix: np.ndarray,
    elo_dict: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Render positive signals and warnings for the match."""
    form_dict = calculate_recent_form_by_matches(df)
    positives_summary = get_all_positive_signals(df, home_team, away_team, elo_dict)
    for entry in positives_summary:
        st.success(f"✅ Pozitivní trendy u {entry['team']}: " + " ".join(entry["messages"]))

    warnings_list, warning_index_home, warning_index_away = get_all_match_warnings(
        df, home_team, away_team, matrix, elo_dict, form_dict
    )

    for w in warnings_list:
        if w["level"] == "high":
            st.error(w["message"])
        elif w["level"] == "medium":
            st.warning(w["message"])
        elif w["level"] == "low":
            st.info(w["message"])

    return warnings_list, warning_index_home, warning_index_away


def display_metrics(
    home_team: str,
    away_team: str,
    xg_home: float,
    xga_home: float,
    xg_away: float,
    xga_away: float,
    xpoints: Dict[str, float],
    btts: Dict[str, float],
    over_under: Dict[str, float],
    outcomes: Dict[str, float],
    confidence_index: float,
    corner_home_exp: float,
    corner_away_exp: float,
    corner_probs: Dict[str, float],
    corner_line: float,
    ml_probs: Optional[Dict[str, float]] = None,
    outcomes_xg: Optional[Dict[str, float]] = None,
    over25_xg: Optional[float] = None,
    secondary_outcomes: Optional[Dict[str, float]] = None,
    secondary_label: str = "Random Forest",
) -> None:
    """Display key statistical metrics and outcome probabilities."""
    st.markdown("## 📊 Klíčové metriky")

    cols = responsive_columns(3)
    cols[0].metric("xG/xGA sezóna",
                   f"{xg_home:.1f}/{xga_home:.1f} vs {xg_away:.1f}/{xga_away:.1f}")
    cols[1].metric("Oček. body (xP)",
                   f"{xpoints['Home xP']:.1f} vs {xpoints['Away xP']:.1f}")
    cols[2].metric("BTTS", f"{btts['BTTS Yes']:.1f}%")
    cols[2].caption(f"Kurzy: {1 / (btts['BTTS Yes'] / 100):.2f}")

    cols = responsive_columns(3)
    cols[0].metric(
        "Over 1.5 / 2.5 / 3.5",
        f"{over_under['Over 1.5']:.1f}% / "
        f"{over_under['Over 2.5']:.1f}% / "
        f"{over_under['Over 3.5']:.1f}%"
    )
    cols[0].caption(
        f"Kurzy: {1 / (over_under['Over 1.5'] / 100):.2f} / "
        f"{1 / (over_under['Over 2.5'] / 100):.2f} / "
        f"{1 / (over_under['Over 3.5'] / 100):.2f}"
    )
    cols[1].metric("Průměrné rohy", f"{corner_home_exp:.1f} vs {corner_away_exp:.1f}")
    over_key = f"Over {corner_line}"
    cols[2].metric(over_key,
                   f"{corner_probs[over_key]:.1f}%",
                   f"{1 / (corner_probs[over_key] / 100):.2f}")
    cols[2].caption(f"Under: {corner_probs[f'Under {corner_line}']:.1f}%")

    if ml_probs:
        cols = responsive_columns(1)
        best = max(ml_probs, key=ml_probs.get)
        cols[0].metric(
            "ML prediction",
            best,
            f"{ml_probs[best]:.1f}%"
        )

    st.markdown("## 🧠 Pravděpodobnosti výsledků")
    cols2 = responsive_columns(4)
    cols2[0].metric("🏠 Výhra domácích",
                    f"{outcomes['Home Win']:.1f}%",
                    f"{1 / (outcomes['Home Win'] / 100):.2f}")
    cols2[1].metric("🤝 Remíza",
                    f"{outcomes['Draw']:.1f}%",
                    f"{1 / (outcomes['Draw'] / 100):.2f}")
    cols2[2].metric("🚶‍♂️ Výhra hostů",
                    f"{outcomes['Away Win']:.1f}%",
                    f"{1 / (outcomes['Away Win'] / 100):.2f}")
    cols2[3].metric("🔒 Confidence", f"{confidence_index:.1f} %")

    if outcomes_xg:
        cols3 = responsive_columns(4)
        cols3[0].metric("🏠 Výhra domácích (xG)",
                        f"{outcomes_xg['Home Win']:.1f}%",
                        f"{1 / (outcomes_xg['Home Win'] / 100):.2f}")
        cols3[1].metric("🤝 Remíza (xG)",
                        f"{outcomes_xg['Draw']:.1f}%",
                        f"{1 / (outcomes_xg['Draw'] / 100):.2f}")
        cols3[2].metric("🚶‍♂️ Výhra hostů (xG)",
                        f"{outcomes_xg['Away Win']:.1f}%",
                        f"{1 / (outcomes_xg['Away Win'] / 100):.2f}")
        if over25_xg is not None:
            cols3[3].metric("Over 2.5 (xG)",
                            f"{over25_xg:.1f}%",
                            f"{1 / (over25_xg / 100):.2f}")
        else:
            cols3[3].markdown(" ")

    if secondary_outcomes:
        st.markdown(f"### {secondary_label} model")
        cols_rf = responsive_columns(3)
        cols_rf[0].metric("🏠 Výhra domácích", f"{secondary_outcomes['Home Win']:.1f}%")
        cols_rf[1].metric("🤝 Remíza", f"{secondary_outcomes['Draw']:.1f}%")
        cols_rf[2].metric("🚶‍♂️ Výhra hostů", f"{secondary_outcomes['Away Win']:.1f}%")







def render_single_match_prediction(
    df,
    season_df,
    full_df,
    home_team,
    away_team,
    league_name,
    gii_dict,
    elo_dict,
):
    st.header(f"🔮 {home_team} vs {away_team}")
    xg_df = load_upcoming_xg()
    xg_row = lookup_xg_row(xg_df, home_team, away_team)
    if xg_row is not None:
        xg_pred = make_poisson_from_xg(xg_row)
        outcomes_xg = xg_pred["outcomes"]
        over25_xg = xg_pred["over_under"].get("Over 2.5")
    else:
        outcomes_xg = None
        over25_xg = None
        st.info("xG data are not available for this matchup.")


    try:
        inputs = compute_match_inputs(df, season_df, home_team, away_team, gii_dict, elo_dict)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    home_exp = inputs["home_exp"]
    away_exp = inputs["away_exp"]
    matrix = inputs["matrix"]
    outcomes = inputs["outcomes"]
    over_under = inputs["over_under"]
    btts = inputs["btts"]
    xpoints = inputs["xpoints"]
    xg_home = inputs["xg_home"]
    xga_home = inputs["xga_home"]
    xg_away = inputs["xg_away"]
    xga_away = inputs["xga_away"]
    expected_tempo = inputs["expected_tempo"]
    tempo_home = inputs["tempo_home"]
    tempo_away = inputs["tempo_away"]
    expected_gii = inputs["expected_gii"]
    corner_home_exp = inputs["corner_home_exp"]
    corner_away_exp = inputs["corner_away_exp"]

    col1, col2 = responsive_columns(2)
    with col1:
        st.markdown("### ⚽ Očekávané skóre")
        st.markdown(
            f"<h4 style='margin-top: -10px; font-size: 24px;'>"
            f"<span style='color:green'>{home_team}</span> {round(home_exp, 1)} : {round(away_exp, 1)} "
            f"<span style='color:green'>{away_team}</span>"
            f"</h4>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### 🎭 Očekávaný styl zápasu")
        col2.markdown(tempo_tag(expected_tempo), unsafe_allow_html=True)

    warnings_list, warning_index_home, warning_index_away = render_warnings(
        df, home_team, away_team, matrix, elo_dict
    )

    form_dog_positive = len(
        detect_positive_factors(
            df,
            away_team if outcomes["Home Win"] > outcomes["Away Win"] else home_team,
            elo_dict,
        )[0]
    ) > 0

    tempo_home_val = tempo_home["tempo"]
    tempo_away_val = tempo_away["tempo"]

    contrarian_score = calculate_contrarian_risk_score(
        matrix,
        home_exp + away_exp,
        tempo_home_val,
        tempo_away_val,
        warning_index_home,
        warning_index_away,
    )
    upset_score = calculate_upset_risk_score(
        outcomes, warning_index_home, warning_index_away, form_dog_positive
    )

    _, pos_score_home = detect_positive_factors(df, home_team, elo_dict)
    _, pos_score_away = detect_positive_factors(df, away_team, elo_dict)
    _, _, form_stability_home = detect_overperformance_and_momentum(df, home_team)
    _, _, form_stability_away = detect_overperformance_and_momentum(df, away_team)
    form_stability_score = (form_stability_home + form_stability_away) / 2
    variance_flag = scoreline_variance_warning(matrix) is not None
    form_stability_score = 1.0  # Pokud nemáš metodu, klidně ponech 1.0

    confidence_index = calculate_confidence_index(
        outcomes=outcomes,
        poisson_matrix=matrix,
        warning_home=warning_index_home,
        warning_away=warning_index_away,
        form_stability_score=form_stability_score,
        pos_home=pos_score_home,
        pos_away=pos_score_away,
        variance_warning=variance_flag,
    )

    corner_line = st.sidebar.slider("Rohová hranice", 5.5, 15.5, 9.5, 0.5)
    corner_matrix = poisson_corner_matrix(corner_home_exp, corner_away_exp)
    corner_probs = corner_over_under_prob(corner_matrix, corner_line)

    ml_features = construct_features_for_match(df, home_team, away_team, elo_dict)
    ml_probs = predict_proba(
        ml_features,
        model_data=(RF_MODEL, RF_FEATURE_NAMES, RF_LABEL_ENCODER),
    )
    use_rf = st.sidebar.toggle("Use Random Forest probabilities", False)
    primary_outcomes = ml_probs if use_rf else outcomes
    secondary_outcomes = outcomes if use_rf else ml_probs
    secondary_label = "Poisson" if use_rf else "Random Forest"

    display_metrics(
        home_team,
        away_team,
        xg_home,
        xga_home,
        xg_away,
        xga_away,
        xpoints,
        btts,
        over_under,
        primary_outcomes,
        confidence_index,
        corner_home_exp,
        corner_away_exp,
        corner_probs,
        corner_line,
        ml_probs,
        outcomes_xg,
        over25_xg,
        secondary_outcomes,
        secondary_label,
    )

    with st.form("bet_form"):
        bet_type = st.selectbox(
            "Bet type",
            [
                f"{home_team} win",
                f"{away_team} win",
                "Draw",
                f"Double {home_team}",
                f"Double {away_team}",
                f"{home_team} w/o draw",
                f"{away_team} w/o draw",
                "Over 1.5",
                "Over 2.5",
                "Over 3.5",
                "Under 2.5",
                "Under 3.5",
                f"{home_team} and over 1.5",
                f"{away_team} and over 1.5",
                f"{home_team} and over 2.5",
                f"{away_team} and over 2.5",
                f"{home_team} -1.0",
                f"{away_team} -1.0",
            ],
        )
        odds = st.number_input("Odds", min_value=1.0, value=1.5, step=0.01)
        stake = st.number_input("Stake", min_value=0.0, value=100.0, step=0.1)
        if st.form_submit_button("Save bet"):
            bet_db.insert_bet(
                league=league_name,
                home_team=home_team,
                away_team=away_team,
                bet_type=bet_type,
                odds=odds,
                stake=stake,
            )
            st.success("Bet saved")

    # Styl hry
    st.markdown("## 🎮 Styl hry")

    cols = responsive_columns(2)
    for i, (team, tempo) in enumerate([(home_team, tempo_home), (away_team, tempo_away)]):
        over, momentum, _= detect_overperformance_and_momentum(df, team)
        team_status = classify_team_strength(df, team)
        strength_emoji = classify_team_strength(df, team)
        strength_text = {
            "Silní":"💪",
            "Průměrní":"⚖️",
            "Slabí":"🪶"
        }.get(strength_emoji, "Neurčití")

        strength_display = f"{strength_text} {strength_emoji}"

        with cols[i]:
            st.markdown(f"### {'🏠' if i == 0 else '🚶‍♂️'} {team}")
            left_col, right_col = responsive_columns(2)

            with left_col:
                st.markdown(f"<p style='font-size:15px'>⚡ Tempo zápasu:</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:20px'>{tempo['rating']}</p>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:15px'>🥾 Tvrdost zápasu:</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:20px'>{tempo['aggressiveness_rating']}</p>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:15px'>📊 Dominance v zápase:</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:20px'>{tempo['imbalance_type']}</p>", unsafe_allow_html=True)

            with right_col:
                st.markdown(f"<p style='font-size:15px'>🧾 Status týmu:</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:20px'>{strength_display}</p>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:15px'>🎯 Overperformance:</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:20px'>{over}</p>", unsafe_allow_html=True)

                st.markdown(f"<p style='font-size:15px'>📈 Momentum:</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:20px'>{momentum}</p>", unsafe_allow_html=True)

    st.markdown("## 🎲 Riziko nečekaného průběhu")
    col1, col2 = responsive_columns(2)
    col1.markdown(colored_risk_tag("🎭 Přestřelka místo nudy", contrarian_score), unsafe_allow_html=True)
    col2.markdown(colored_risk_tag("🧨 Překvapení outsidera", upset_score), unsafe_allow_html=True)

    #df_team_status = display_team_status_table(home_team, away_team, df, elo_dict)
    # st.markdown("## 📊 Porovnání týmů")
    # st.dataframe(display_team_status_table(home_team, away_team, df, elo_dict), use_container_width=True, hide_index=True)

    # Týmové statistiky
    st.markdown("## 🧠 Očekávané týmové statistiky")
    stat_map = {
        'Střely': ('HS', 'AS'),
        'Střely na branku': ('HST', 'AST'),
        'Rohy': ('HC', 'AC'),
        'Žluté karty': ('HY', 'AY')
    }
    elo_stats = expected_team_stats_weighted_by_elo(df, home_team, away_team, stat_map, elo_dict)
    for stat, values in elo_stats.items():
        st.markdown(f"- **{stat}**: `{home_team}` {values['Home']} – {values['Away']} `{away_team}`")

    # Výkon podle typu soupeřů
    st.markdown("## 🏟️ Výkon dle typu soupeřů (Doma / Venku)")
    strength_home = classify_team_strength(df, home_team)
    strength_away = classify_team_strength(df, away_team)

    def display_merged_table(data, team_name, teamstrength):
        emoji_map = {"Silní": "💪", "Průměrní": "⚖️", "Slabí": "🪶"}
        icon = emoji_map.get(teamstrength, "")
        st.markdown(f"### {team_name} {icon} ")
        
        df_disp = pd.DataFrame(data).T
        df_disp = df_disp[
            [
                "Zápasy",
                "Góly",
                "Obdržené",
                "Střely",
                "Na branku",
                "xG",
                "xGA",
                "Body/zápas",
                "Čistá konta %",
            ]
        ]
        st.dataframe(df_disp, use_container_width=True)

    display_merged_table(
        merged_home_away_opponent_form(full_df, home_team),
        home_team,
        strength_home,
    )
    display_merged_table(
        merged_home_away_opponent_form(full_df, away_team),
        away_team,
        strength_away,
    )

    # Head-to-head statistiky
    # Head-to-Head – kompaktní přehled
    st.markdown("## 💬 Head-to-Head")
    h2h = get_head_to_head_stats(full_df, home_team, away_team, last_n=None)

    if h2h:
        h2h_cols = responsive_columns(6)

        h2h_cols[0].markdown("🆚 **Zápasy**")
        h2h_cols[0].markdown(f"<h3 style='margin-top:-5px'>{h2h['matches']}</h3>", unsafe_allow_html=True)

        h2h_cols[1].markdown(f"✅ **{home_team} výher**")
        h2h_cols[1].markdown(f"<h3 style='margin-top:-5px'>{h2h['home_wins']}</h3>", unsafe_allow_html=True)

        h2h_cols[2].markdown("🤝 **Remízy**")
        h2h_cols[2].markdown(f"<h3 style='margin-top:-5px'>{h2h['draws']}</h3>", unsafe_allow_html=True)

        h2h_cols[3].markdown(f"✅ **{away_team} výher**")
        h2h_cols[3].markdown(f"<h3 style='margin-top:-5px'>{h2h['away_wins']}</h3>", unsafe_allow_html=True)

        h2h_cols[4].markdown("🎯 **Průměr gólů**")
        h2h_cols[4].markdown(f"<h3 style='margin-top:-5px'>{h2h['avg_goals']}</h3>", unsafe_allow_html=True)

        h2h_cols[5].markdown("🤝 **BTTS / Over 2.5**")
        h2h_cols[5].markdown(f"<h3 style='margin-top:-5px'>{h2h['btts_pct']}% / {h2h['over25_pct']}%</h3>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Nenašly se žádné vzájemné zápasy.")

        
    # Nejpravděpodobnější výsledky
    top_scores = get_top_scorelines(matrix, top_n=5)
    home_probs, away_probs = get_goal_probabilities(matrix)
    goly = [1, 2, 3, 4, 5]
    goal_chances = pd.DataFrame({
        "Góly": goly,
        home_team: [f"{round(home_probs[i]*100, 1)} %" for i in goly],
        away_team: [f"{round(away_probs[i]*100, 1)} %" for i in goly],
    })
    top_df = pd.DataFrame([
        {"Skóre": f"{a}:{b}", "Pravděpodobnost": f"{round(p*100, 1)} %"}
        for (a, b), p in top_scores
    ])

    col1, col2 = responsive_columns(2)
    col1.markdown("### 🏅 Nejpravděpodobnější výsledky")
    col1.dataframe(top_df, use_container_width=True, hide_index=True)
    col2.markdown("### 🎯 Šance na počet vstřelených gólů")
    col2.dataframe(goal_chances, use_container_width=True, hide_index=True)


    
    style_home = get_team_style_vs_opponent_type(full_df, home_team, away_team)
    style_away = get_team_style_vs_opponent_type(full_df, away_team, home_team)
    st.divider()
    radar_cols = st.columns(4)
    if isinstance(style_home, Mapping) and style_home:
        radar_cols[0].plotly_chart(
            plot_style_radar(style_home), use_container_width=True
        )
        radar_cols[0].caption(
            " | ".join(TEAM_COMPARISON_DESC_MAP.get(k, k) for k in style_home)
        )
    if isinstance(style_away, Mapping) and style_away:
        radar_cols[1].plotly_chart(
            plot_style_radar(style_away), use_container_width=True
        )
        radar_cols[1].caption(
            " | ".join(TEAM_COMPARISON_DESC_MAP.get(k, k) for k in style_away)
        )

    # Extrakce konkrétních warningů z výstupu
    variance_warning_msg = next((w["message"] for w in warnings_list if "rozptyl" in w["message"].lower()), None)
    style_form_warning_msg = next((w["message"] for w in warnings_list if "forma" in w["message"].lower()), None)
    style_conflict_warning_msg = next((w["message"] for w in warnings_list if "styl" in w["message"].lower()), None)

    
    # Export Excel
    
    excel_file = generate_excel_analysis_export(
            league_name, home_team, away_team,
            (home_exp, away_exp), outcomes, over_under, btts,
            xpoints, xg_home, xg_away,
            expected_tempo, expected_gii,
            warnings_home=calculate_warning_index(df, home_team, elo_dict)[0],
            warnings_away=calculate_warning_index(df, away_team, elo_dict)[0],
            positives_home=detect_positive_factors(df, home_team, elo_dict)[0],
            positives_away=detect_positive_factors(df, away_team, elo_dict)[0],
            team_stats=elo_stats,
            style_home=tempo_home, style_away=tempo_away,
            form_home=pd.DataFrame(merged_home_away_opponent_form(df, home_team)).T,
            form_away=pd.DataFrame(merged_home_away_opponent_form(df, away_team)).T,
            h2h_stats=h2h,
            top_scorelines=top_scores,
            goal_probs=goal_chances,
            variance_warning=variance_warning_msg,
            style_form_warning=style_form_warning_msg,
            style_conflict_warning=style_conflict_warning_msg,
        )    
    st.download_button(
    "📥 Stáhnout analytickou zprávu jako Excel",
        data=excel_file,
        file_name=f"{home_team}_vs_{away_team}_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

