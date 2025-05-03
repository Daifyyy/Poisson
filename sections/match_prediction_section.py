import streamlit as st
import pandas as pd
from utils.poisson_utils import (
    calculate_elo_ratings, expected_goals_weighted_by_elo, poisson_prediction, match_outcomes_prob,
    over_under_prob, btts_prob, calculate_expected_points, calculate_pseudo_xg_for_team,
    analyze_opponent_strength, expected_match_style_score, intensity_score_to_emoji,
    expected_match_tempo, tempo_to_emoji, get_top_scorelines, get_goal_probabilities,
    detect_risk_factors, detect_positive_factors, calculate_warning_index,
    expected_team_stats_weighted_by_elo, classify_team_strength, merged_home_away_opponent_form,
    get_head_to_head_stats, calculate_match_tempo
)
from utils.frontend_utils import display_team_status_table
from utils.poisson_utils.match_style import tempo_tag




def render_single_match_prediction(df, season_df, home_team, away_team, league_name, gii_dict):
    st.header(f"🔮 {home_team} vs {away_team}")

    try:
        elo_dict = calculate_elo_ratings(df)
        home_exp, away_exp = expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    matrix = poisson_prediction(home_exp, away_exp)
    outcomes = match_outcomes_prob(matrix)
    over_under = over_under_prob(matrix)
    btts = btts_prob(matrix)
    xpoints = calculate_expected_points(outcomes)

    xg_home = calculate_pseudo_xg_for_team(season_df, home_team)
    xg_away = calculate_pseudo_xg_for_team(season_df, away_team)

    col1, col2 = st.columns(2)
    expected_gii = round((gii_dict.get(home_team, 0) + gii_dict.get(away_team, 0)) / 2, 2)
    expected_gii_emoji = intensity_score_to_emoji(expected_gii)
    expected_tempo = expected_match_tempo(
        df,            # DataFrame se zápasy
        home_team,     # Domácí tým
        away_team,     # Hostující tým
        elo_dict,      # Slovník s ELO hodnotami
        home_exp,      # Očekávané góly domácích (např. z Poissona)
        away_exp,      # Očekávané góly hostů
        xg_home["xG_home"],      # Pseudo-xG domácích
        xg_away["xG_away"]       # Pseudo-xG hostů
    )

    tempo_emoji = tempo_to_emoji(expected_tempo)

    with col1:
        st.markdown("### ⚽ Očekávané skóre")
        st.markdown(
            f"<h4 style='margin-top: -10px; font-size: 24px;'>"
            f"<span style='color:green'>{home_team}</span> {round(home_exp, 1)} : {round(away_exp, 1)} "
            f"<span style='color:green'>{away_team}</span>"
            f"</h4>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### 🎭 Očekávaný styl zápasu")
        # col2.markdown(f"### {tempo_emoji} ({expected_tempo})")
        col2.markdown(tempo_tag(expected_tempo), unsafe_allow_html=True)


    # Klíčové metriky
    st.markdown("## 📊 Klíčové metriky")
    cols = st.columns(4)
    cols[0].metric("xG sezóna", f"{xg_home['xG_home']} vs {xg_away['xG_away']}")
    cols[1].metric("Oček. body (xP)", f"{xpoints['Home xP']} vs {xpoints['Away xP']}")
    cols[2].metric("BTTS / Over 2.5", f"{btts['BTTS Yes']}% / {over_under['Over 2.5']}%")
    cols[2].caption(f"Kurzy: {1 / (btts['BTTS Yes'] / 100):.2f} / {1 / (over_under['Over 2.5'] / 100):.2f}")

    # Výsledkové pravděpodobnosti
    st.markdown("## 🧠 Pravděpodobnosti výsledků")
    cols2 = st.columns(3)
    cols2[0].metric("🏠 Výhra domácích", f"{outcomes['Home Win']}%", f"{1 / (outcomes['Home Win'] / 100):.2f}")
    cols2[1].metric("🤝 Remíza", f"{outcomes['Draw']}%", f"{1 / (outcomes['Draw'] / 100):.2f}")
    cols2[2].metric("🚶‍♂️ Výhra hostů", f"{outcomes['Away Win']}%", f"{1 / (outcomes['Away Win'] / 100):.2f}")

    # for team in [home_team, away_team]:
    #     warnings, _ = detect_risk_factors(df, team, elo_dict)
    #     if warnings:
    #         st.warning(f"⚠️ Rizika pro {team}: " + " ".join(warnings))

    for team in [home_team, away_team]:
        positives, _ = detect_positive_factors(df, team, elo_dict)
        if positives:
            st.success(f"✅ Pozitivní trendy u {team}: " + " ".join(positives))

    for team in [home_team, away_team]:
        warnings, warning_score = calculate_warning_index(df, team, elo_dict)
        if warnings:
            st.error(f"⚠️ {team} Warning Index: {int(warning_score * 100)}% - " + ", ".join(warnings))

    #df_team_status = display_team_status_table(home_team, away_team, df, elo_dict)
    st.markdown("## 📊 Porovnání týmů")
    st.write(display_team_status_table(home_team, away_team, df, elo_dict), use_container_width=True, hide_index=True)



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
        df_disp = df_disp[["Zápasy", "Góly", "Obdržené", "Střely", "Na branku", "xG", "Body/zápas", "Čistá konta %"]]
        st.dataframe(df_disp)

    




    display_merged_table(merged_home_away_opponent_form(df, home_team), home_team, strength_home)
    display_merged_table(merged_home_away_opponent_form(df, away_team), away_team, strength_away)

    # Head-to-head statistiky
    # Head-to-Head – kompaktní přehled
    st.markdown("## 💬 Head-to-Head")
    h2h = get_head_to_head_stats(df, home_team, away_team)

    if h2h:
        h2h_cols = st.columns(6)

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


    # Styl hry
    st.markdown("## 🎮 Styl hry")
    tempo_home = calculate_match_tempo(df, home_team, elo_dict.get(away_team, 1500), True, elo_dict)
    tempo_away = calculate_match_tempo(df, away_team, elo_dict.get(home_team, 1500), False, elo_dict)

    cols = st.columns(2)
    for i, (team, tempo) in enumerate([(home_team, tempo_home), (away_team, tempo_away)]):
        with cols[i]:
            st.markdown(f"### {'🏠' if i == 0 else '🚶‍♂️'} {team}")
            st.markdown(f"<p style='font-size:15px'>⚡ Tempo zápasu:</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px'>{tempo['rating']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:15px'>🥾 Tvrdost zápasu:</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px'>{tempo['aggressiveness_rating']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:15px'>📊 Dominance v zápase:</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:20px'>{tempo['imbalance_type']}</p>", unsafe_allow_html=True)

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

    col1, col2 = st.columns(2)
    col1.markdown("### 🏅 Nejpravděpodobnější výsledky")
    col1.dataframe(top_df, use_container_width=True, hide_index=True)
    col2.markdown("### 🎯 Šance na počet vstřelených gólů")
    col2.dataframe(goal_chances, use_container_width=True, hide_index=True)
