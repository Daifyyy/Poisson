import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import date
#from flask import Flask, request, jsonify
from utils.poisson_utils import (
    load_data, validate_dataset, calculate_team_strengths,get_team_average_gii,
    get_head_to_head_stats,classify_team_strength,get_team_average_gii,get_top_scorelines,
    poisson_prediction, match_outcomes_prob, over_under_prob,
    btts_prob, prob_to_odds,calculate_pseudo_xg,calculate_match_tempo,calculate_gii_zscore,
    analyze_opponent_strength, calculate_expected_points,classify_team_strength,intensity_score_to_emoji,
    expected_goals_weighted_by_elo,expected_match_style_score,get_goal_probabilities,
    calculate_elo_ratings, calculate_recent_form, detect_current_season,expected_team_stats_weighted_by_elo,
    calculate_team_pseudo_xg,calculate_expected_and_actual_points,merged_home_away_opponent_form,detect_risk_factors,detect_positive_factors,
    display_team_status_table,calculate_warning_index,detect_overperformance_and_momentum
)



st.set_page_config(page_title="⚽ Poisson Predictor", layout="wide")
#st.title("⚽ Poisson Match Predictor")

# Výběr ligy a načtení dat
league_files = {
    "E0 (Premier League)": "data/E0_combined_full_updated.csv",
    "E1 (Championship)": "data/E1_combined_full_updated.csv",  
    "SP1 (La Liga)": "data/SP1_combined_full_updated.csv",
    "B1 (Jupiler League)": "data/B1_combined_full_updated.csv",  # Nová liga
    "D1 (Bundesliga)": "data/D1_combined_full_updated.csv",  
    "D2 (2. Bundesliga)": "data/D2_combined_full_updated.csv",# Nová liga
    "I1 (Seria A)": "data/I1_combined_full_updated.csv",   
    "F1 (Ligue 1)": "data/F1_combined_full_updated.csv",  # Nová liga
    "N1 (Eredivisie)": "data/N1_combined_full_updated.csv",  # Nová liga      # Nová liga
    "P1 (Primeira Liga)": "data/P1_combined_full_updated.csv",  # Nová liga
    "T1 (Super League)": "data/T1_combined_full_updated.csv",  # Nová liga
}
league_name = st.sidebar.selectbox("🌍 Vyber ligu", list(league_files.keys()))
league_file = league_files[league_name]

df = load_data(league_file)
validate_dataset(df)
season_df, season_start = detect_current_season(df)
team_strengths, _, _ = calculate_team_strengths(df)
season_df = calculate_gii_zscore(season_df)
gii_dict = get_team_average_gii(season_df)

if "match_list" not in st.session_state:
    st.session_state.match_list = []

# print(season_df)
# print(season_start)
# Výběr týmů
st.sidebar.header("🏟️ Zápas")
teams_in_season = sorted(set(season_df["HomeTeam"].unique()) | set(season_df["AwayTeam"].unique()))
home_team = st.sidebar.selectbox("Domácí tým", teams_in_season)
away_team = st.sidebar.selectbox("Hostující tým", teams_in_season)
multi_prediction_mode = st.sidebar.checkbox("📝 Hromadné predikce")
gii_home = gii_dict.get(home_team, 0)
gii_away = gii_dict.get(away_team, 0)
expected_gii = round((gii_home + gii_away) / 2, 2)

if home_team == away_team:
    st.header(f"🏆 {league_name}")

    # Základní metriky
    num_matches = len(season_df)
    avg_goals = round((season_df['FTHG'] + season_df['FTAG']).mean(), 1)
    season_df['BTTS'] = season_df.apply(lambda row: int(row['FTHG'] > 0 and row['FTAG'] > 0), axis=1)
    btts_pct = round(100 * season_df['BTTS'].mean(), 1)
    over_25 = round(100 * season_df[(season_df['FTHG'] + season_df['FTAG']) > 2.5].shape[0] / num_matches, 1)
    st.markdown(f"📅 Zápasů: {num_matches} ⚽ Průměr gólů: {avg_goals} 🥅 BTTS: {btts_pct}% 📈 Over 2.5: {over_25}%")

    # ELO, forma, body
    elo_dict = calculate_elo_ratings(season_df)
    form_dict = calculate_recent_form(season_df, days=31)
    points_data = calculate_expected_and_actual_points(season_df)

    # Přehled tabulky
    team_stats = season_df.groupby("HomeTeam").agg({
        "FTHG": "mean", "FTAG": "mean", "HS": "mean", "HST": "mean", "HC": "mean", "HY": "mean"
    }).rename(columns={
        "FTHG": "Góly doma", "FTAG": "Góly venku", "HS": "Střely", "HST": "Na branku", "HC": "Rohy", "HY": "Žluté"
    })

    shots_on_target = season_df.groupby("HomeTeam")["HST"].mean()
    shots_on_target_away = season_df.groupby("AwayTeam")["AST"].mean()

    combined_sot = pd.DataFrame({
        "Na branku doma": shots_on_target,
        "Na branku venku": shots_on_target_away
    })
    combined_sot["Celkem na branku"] = combined_sot["Na branku doma"] + combined_sot["Na branku venku"]

    
    over25 = season_df.groupby("HomeTeam").apply(lambda x: (x['FTHG'] + x['FTAG'] > 2.5).mean() * 100).round(0)
    btts = season_df.groupby("HomeTeam")["BTTS"].mean().mul(100).round(0)
    xg_stats = calculate_team_pseudo_xg(season_df)

    # Pomocná funkce pro čistá konta
    def calculate_clean_sheets(team, df):
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
        cs = 0
        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team and row['FTAG'] == 0:
                cs += 1
            elif row['AwayTeam'] == team and row['FTHG'] == 0:
                cs += 1
        return round(100 * cs / len(team_matches), 1) if len(team_matches) > 0 else 0


    # Upravený summary_table s novými metrikami
    summary_table = pd.DataFrame({
        "Tým": team_stats.index,
        "Elo": team_stats.index.map(lambda t: elo_dict.get(t, 1500)),
        "Body": team_stats.index.map(lambda t: points_data.get(t, {}).get("points", 0)),
        "Form": team_stats.index.map(lambda t: 
            "🔥🔥🔥" if form_dict.get(t, 0) > 2.5 else
            "🔥🔥" if form_dict.get(t, 0) > 2.0 else
            "🔥" if form_dict.get(t, 0) > 1.5 else
            "💤" if form_dict.get(t, 0) > 1.2 else
            "❄️" if form_dict.get(t, 0) > 0.7 else
            "❄️❄️" if form_dict.get(t, 0) > 0.5 else
            "❄️❄️❄️"
        ),
        "Góly/zápas": ((team_stats["Góly doma"] + team_stats["Góly venku"]) / 2).round(2),
        "Intenzita": team_stats.index.map(lambda t: intensity_score_to_emoji(gii_dict.get(t, 0))),
        #"GII": team_stats.index.map(lambda t: gii_dict.get(t, 0)),
        "Čistá konta %": team_stats.index.map(lambda t: calculate_clean_sheets(t, season_df)),
        "Over 2.5 %": team_stats.index.map(over25).astype(str) + "%",
        "BTTS %": team_stats.index.map(btts).astype(str) + "%"
    }).sort_values("Elo", ascending=False).reset_index(drop=True)

    st.dataframe(summary_table, hide_index=True)

    # Top 5 přehled – různé kategorie
    st.markdown("### 🌟 Top 5 týmy")
    cols = st.columns(4)
    cols[0].markdown("🔮 **Nejvíc gólů**")
    cols[0].dataframe(summary_table.sort_values("Góly/zápas", ascending=False).head(5)[["Tým", "Góly/zápas"]], hide_index=True)

    # Nová sekce: Nejvíce obdržených gólů
    conceded_stats = []
    for team in season_df['HomeTeam'].unique():
        home = season_df[season_df['AwayTeam'] == team]
        away = season_df[season_df['HomeTeam'] == team]
        goals_against = pd.concat([home['FTHG'], away['FTAG']]).mean()
        conceded_stats.append({"Tým": team, "Obdržené góly": round(goals_against, 2)})
    conceded_df = pd.DataFrame(conceded_stats).sort_values("Obdržené góly", ascending=False).reset_index(drop=True)
    cols[1].markdown("🔴 **Nejvíce obdržených gólů**")
    cols[1].dataframe(conceded_df.head(5), hide_index=True)

    # 📉 Nejhorší forma (správně seřazená + emoji)
    form_stats = []
    for team in season_df['HomeTeam'].unique():
        recent_matches = season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)].sort_values("Date").tail(5)
        total_points = 0
        for _, row in recent_matches.iterrows():
            if row['HomeTeam'] == team:
                total_points += 3 if row['FTHG'] > row['FTAG'] else 1 if row['FTHG'] == row['FTAG'] else 0
            elif row['AwayTeam'] == team:
                total_points += 3 if row['FTAG'] > row['FTHG'] else 1 if row['FTAG'] == row['FTHG'] else 0
        avg_points = total_points / 5
        form_stats.append({"Tým": team, "Body/zápas": avg_points})

    form_df = pd.DataFrame(form_stats).sort_values("Body/zápas").reset_index(drop=True)

    def form_to_emoji(avg_points):
        if avg_points > 2.5:
            return "🔥🔥🔥"
        elif avg_points > 2.0:
            return "🔥🔥"
        elif avg_points > 1.5:
            return "🔥"
        elif avg_points > 1.2:
            return "💤"
        elif avg_points > 0.7:
            return "❄️"
        elif avg_points > 0.5:
            return "❄️❄️"
        else:
            return "❄️❄️❄️"

    form_df["Form"] = form_df["Body/zápas"].apply(form_to_emoji)
    cols[2].markdown("📉 **Nejhorší forma**")
    cols[2].dataframe(form_df[["Tým", "Form"]].head(5), hide_index=True)


    cols[3].markdown("📈 **Nejlepší forma**")
    cols[3].dataframe(summary_table.sort_values("Form", ascending=False).head(5)[["Tým", "Form"]], hide_index=True)

    # Další 4-sloupcový blok – ELO + styl
    st.markdown("### 🔧 Styl a vývoj týmů")
    elo_start = calculate_elo_ratings(season_df[season_df['Date'] <= season_df['Date'].min() + pd.Timedelta(days=5)])
    elo_end = calculate_elo_ratings(season_df)
    elo_change = [{"Tým": t, "Změna": round(elo_end.get(t, 1500) - elo_start.get(t, 1500), 1)} for t in elo_end]
    elo_df = pd.DataFrame(elo_change).sort_values("Změna", ascending=False).reset_index(drop=True)
    elo_drop_df = elo_df.sort_values("Změna").head(5).reset_index(drop=True)

    
        
    # Styl hry (ofenzivní/defenzivní)
    offensive_style = []
    defensive_style = []
    for team in season_df['HomeTeam'].unique():
        home = season_df[season_df['HomeTeam'] == team]
        away = season_df[season_df['AwayTeam'] == team]
        shots = pd.concat([home['HS'], away['AS']]).mean()
        sot = pd.concat([home['HST'], away['AST']]).mean()
        corners = pd.concat([home['HC'], away['AC']]).mean()
        xg = (sot * (pd.concat([home['FTHG'], away['FTAG']]).sum() / sot)) if sot > 0 else 0
        fouls = pd.concat([home['HF'], away['AF']]).mean()
        offensive_index = shots * 0.25 + sot * 0.25 + corners * 0.2 + xg * 0.2 + fouls * 0.1
        defensive_index = (1 / (shots + 1)) * 0.3 + (1 / (sot + 1)) * 0.25 + (1 / (corners + 1)) * 0.2 + (1 / (xg + 0.1)) * 0.15 + (1 / (fouls + 1)) * 0.1
        offensive_style.append({"Tým": team, "Ofenzivní styl index": round(offensive_index, 2)})
        defensive_style.append({"Tým": team, "Defenzivní styl index": round(defensive_index, 2)})
    off_df = pd.DataFrame(offensive_style).sort_values("Ofenzivní styl index", ascending=False).reset_index(drop=True)
    def_df = pd.DataFrame(defensive_style).sort_values("Defenzivní styl index", ascending=False).reset_index(drop=True)

    cols2 = st.columns(4)
    cols2[0].markdown("📈 **ELO zlepšení**")
    cols2[0].dataframe(elo_df.head(5), hide_index=True)
    cols2[1].markdown("📉 **ELO poklesy**")
    cols2[1].dataframe(elo_drop_df, hide_index=True)
    cols2[2].markdown("⚡ **Ofenzivní styl**")
    cols2[2].dataframe(off_df.head(5)[["Tým", "Ofenzivní styl index"]], hide_index=True)
    cols2[3].markdown("🧱 **Defenzivní styl**")
    cols2[3].dataframe(def_df.head(5)[["Tým", "Defenzivní styl index"]], hide_index=True)

    st.stop()

if not multi_prediction_mode:

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

    xg_home = calculate_pseudo_xg(season_df, home_team)
    xg_away = calculate_pseudo_xg(season_df, away_team)

    strength_home = analyze_opponent_strength(season_df, home_team, is_home=True)
    strength_away = analyze_opponent_strength(season_df, away_team, is_home=False)


    # 🧮 Skóre
    mss_prediction = expected_match_style_score(season_df, home_team, away_team, elo_dict)
    col1, col2 = st.columns(2)
    expected_gii_emoji = intensity_score_to_emoji(expected_gii)

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
        col2.markdown(f"### {expected_gii_emoji}")
        col2.caption(f"Založeno na GII skóre {home_team} ({gii_home}) a {away_team} ({gii_away})")





    # 🔢 Statistiky v řádku
    st.markdown("## 📊 Klíčové metriky")
    cols = st.columns(4)
    cols[0].metric("xG sezóna", f"{xg_home['xG_home']} vs {xg_away['xG_away']}")
    cols[1].metric("Oček. body (xP)", f"{xpoints['Home xP']} vs {xpoints['Away xP']}")
    cols[2].metric("BTTS / Over 2.5", f"{btts['BTTS Yes']}% / {over_under['Over 2.5']}%")
    cols[2].caption(f"Kurzy: {prob_to_odds(btts['BTTS Yes'])} / {prob_to_odds(over_under['Over 2.5'])}")

    # 🧠 Pravděpodobnosti výsledků
    st.markdown("## 🧠 Pravděpodobnosti výsledků")
    cols2 = st.columns(3)
    cols2[0].metric("🏠 Výhra domácích", f"{outcomes['Home Win']}%", f"{prob_to_odds(outcomes['Home Win'])}")
    cols2[1].metric("🤝 Remíza", f"{outcomes['Draw']}%", f"{prob_to_odds(outcomes['Draw'])}")
    cols2[2].metric("🚶‍♂️ Výhra hostů", f"{outcomes['Away Win']}%", f"{prob_to_odds(outcomes['Away Win'])}")
    #RIZIKA
    warnings_home, risk_home = detect_risk_factors(df, home_team, elo_dict)
    warnings_away, risk_away = detect_risk_factors(df, away_team, elo_dict)

    if warnings_home:
        st.warning(f"⚠️ Rizika pro {home_team}: " + " ".join(warnings_home))
    if warnings_away:
        st.warning(f"⚠️ Rizika pro {away_team}: " + " ".join(warnings_away))

    # st.metric(f"Risk skóre {home_team}", f"{risk_home*100:.0f}%")
    # st.metric(f"Risk skóre {away_team}", f"{risk_away*100:.0f}%")

    #POZITIVA   
    positives_home, pos_score_home = detect_positive_factors(df, home_team, elo_dict)
    positives_away, pos_score_away = detect_positive_factors(df, away_team, elo_dict)

    if positives_home:
        st.success(f"✅ Pozitivní trendy u {home_team}: " + " ".join(positives_home))
    if positives_away:
        st.success(f"✅ Pozitivní trendy u {away_team}: " + " ".join(positives_away))

    # st.metric(f"Form Boost {home_team}", f"{pos_score_home*100:.0f}%")
    # st.metric(f"Form Boost {away_team}", f"{pos_score_away*100:.0f}%")

    warnings_home, warning_score_home = calculate_warning_index(df, home_team, elo_dict)
    warnings_away, warning_score_away = calculate_warning_index(df, away_team, elo_dict)

    if warnings_home:
        st.error(f"⚠️ {home_team} Warning Index: {int(warning_score_home * 100)}% - " + ", ".join(warnings_home))
    if warnings_away:
        st.error(f"⚠️ {away_team} Warning Index: {int(warning_score_away * 100)}% - " + ", ".join(warnings_away))


    



    display_team_status_table(home_team, away_team, df, elo_dict)

    overperf_home, momentum_home = detect_overperformance_and_momentum(df, home_team)
    overperf_away, momentum_away = detect_overperformance_and_momentum(df, away_team)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"🏠 **{home_team}**")
        st.markdown(f"- Overperformance: {overperf_home}")
        st.markdown(f"- Momentum: {momentum_home}")

    with col2:
        st.markdown(f"🚶‍♂️ **{away_team}**")
        st.markdown(f"- Overperformance: {overperf_away}")
        st.markdown(f"- Momentum: {momentum_away}")


            
    
    # 📌 Týmové statistiky
    st.markdown("## 🧠 Očekávané týmové statistiky")

    # Načtení ELO ratingů
    elo_dict = calculate_elo_ratings(df)

    # Definice sledovaných statistik
    stat_map = {
        'Střely': ('HS', 'AS'),
        'Střely na branku': ('HST', 'AST'),
        'Rohy': ('HC', 'AC'),
        'Žluté karty': ('HY', 'AY')
    }

    # Výpočet statistik na základě podobnosti soupeře dle ELO
    elo_stats = expected_team_stats_weighted_by_elo(df, home_team, away_team, stat_map, elo_dict)

    # Zobrazení v rozhraní
    for stat, values in elo_stats.items():
        st.markdown(f"- **{stat}**: `{home_team}` {values['Home']} – {values['Away']} `{away_team}`")


    # 📊 Výkon vůči soupeřům (detailní přehled)
    # st.markdown("## ⚖️ Výkon proti typům soupeřů")
    # perf_home = analyze_opponent_strength(df, home_team, is_home=True)
    # perf_away = analyze_opponent_strength(df, away_team, is_home=False)

    # st.markdown(f"### 🏠 Výkon domácího týmu – {home_team}")
    # home_cols = st.columns(3)
    # with st.container():
    #     for i, cat in enumerate(['vs_strong', 'vs_average', 'vs_weak']):
    #         with home_cols[i]:
    #             st.metric("Zápasy", perf_home[cat]['matches'])
    #             st.metric("Góly", perf_home[cat]['goals'])
    #             st.metric("Konverze", f"{perf_home[cat]['con_rate']*100:.1f}%")
    #             st.metric("Body/zápas", perf_home[cat]['xP'])
    #             st.caption(["💪 Silní", "⚖️ Průměrní", "🪶 Slabí"][i])

    # st.markdown(f"### 🚶‍♂️ Výkon hostujícího týmu – {away_team}")
    # away_cols = st.columns(3)
    # with st.container():
    #     for i, cat in enumerate(['vs_strong', 'vs_average', 'vs_weak']):
    #         with away_cols[i]:
    #             st.metric("Zápasy", perf_away[cat]['matches'])
    #             st.metric("Góly", perf_away[cat]['goals'])
    #             st.metric("Konverze", f"{perf_away[cat]['con_rate']*100:.1f}%")
    #             st.metric("Body/zápas", perf_away[cat]['xP'])
    #             st.caption(["💪 Silní", "⚖️ Průměrní", "🪶 Slabí"][i])
                
    st.markdown("## 🏟️ Výkon dle typu soupeřů (Doma / Venku)")
    strength_home = classify_team_strength(df, home_team)
    strength_away = classify_team_strength(df, away_team)
    def display_merged_table(data, team_name, teamstrength):
        st.markdown(f"### {team_name} ({teamstrength})")
        df_disp = pd.DataFrame(data).T  # index = ['💪 Silní', '⚖️ Průměrní', '🪶 Slabí']
        df_disp = df_disp[["Zápasy", "Góly", "Obdržené", "Střely", "Na branku", "xG", "Body/zápas", "Čistá konta %"]]
        st.dataframe(df_disp)

    merged_home = merged_home_away_opponent_form(df, home_team)
    merged_away = merged_home_away_opponent_form(df, away_team)

    display_merged_table(merged_home, home_team,strength_home)
    display_merged_table(merged_away, away_team,strength_away)

    # 🤜 Head-to-Head statistiky
    st.markdown("## 💬 Head-to-Head statistiky")

    h2h = get_head_to_head_stats(df, home_team, away_team)
    if h2h:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Počet zápasů", h2h['matches'])
            st.metric(f"Výhry {home_team}", h2h['home_wins'])
            st.metric("Remízy", h2h['draws'])
            st.metric(f"Výhry {away_team}", h2h['away_wins'])
        with col2:
            st.metric("🎯 Průměr gólů", h2h['avg_goals'])
            st.metric("🤝 BTTS", f"{h2h['btts_pct']} %")
            st.metric("📈 Over 2.5", f"{h2h['over25_pct']} %")
    else:
        st.warning("⚠️ Nenašly se žádné vzájemné zápasy.")



    # 🎮 Styl hry – Tempo & Nerovnováha
    st.markdown("## 🎮 Styl hry")
    # Výpočet metriky tempa a tvrdosti
    tempo_stats_home = calculate_match_tempo(df, home_team, elo_dict.get(away_team, 1500), is_home=True, elo_dict=elo_dict)
    tempo_stats_away = calculate_match_tempo(df, away_team, elo_dict.get(home_team, 1500), is_home=False, elo_dict=elo_dict)

    # 🧱 Sekce pro tempo + tvrdost zápasů
    cols = st.columns(2)

    # with cols[0]:
    #     st.markdown(f"### 🏠 {home_team}")
    #     st.metric("Tempo zápasu", f"{tempo_stats_home['rating']}")
    #     #st.markdown(f"{tempo_stats_home['rating']} ({tempo_stats_home['percentile']} percentil)")

    #     st.metric("Tvrdost zápasu", f"{tempo_stats_home['aggressiveness_rating']}")
    #     #st.markdown(f"{tempo_stats_home['aggressiveness_rating']}")

    #     st.metric("Dominance v zápase", f"{tempo_stats_home['imbalance_type']}")
    #     #st.caption(f"{tempo_stats_home['imbalance_type']} ({tempo_stats_home['imbalance']})")

    # with cols[1]:
    #     st.markdown(f"### 🚶‍♂️ {away_team}")
    #     st.metric("Tempo zápasu", f"{tempo_stats_away['rating']} ")
    #     #t.markdown(f"({tempo_stats_away['percentile']} percentil)")

    #     st.metric("Tvrdost zápasu", f"{tempo_stats_away['aggressiveness_rating']}")
    #     #st.markdown(f"{tempo_stats_away['aggressiveness_rating']}")

    #     st.metric("Dominance v zápase", f"{tempo_stats_away['imbalance_type']}")
    #     #st.markdown(f"{tempo_stats_away['aggressiveness_rating']}")
    #     #st.caption(f"{tempo_stats_away['imbalance_type']} ({tempo_stats_away['imbalance']})")


    # Sekce pro tempo + tvrdost zápasů
    cols = st.columns(2)

    with cols[0]:
        st.markdown(f"### 🏠 {home_team}")
        st.markdown(f"<p style='font-size:15px'>⚡ Tempo zápasu:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px'>{tempo_stats_home['rating']}</p>", unsafe_allow_html=True)

        st.markdown(f"<p style='font-size:15px'>🥾 Tvrdost zápasu:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px'>{tempo_stats_home['aggressiveness_rating']}</p>", unsafe_allow_html=True)

        st.markdown(f"<p style='font-size:15px'>📊 Dominance v zápase:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px'>{tempo_stats_home['imbalance_type']}</p>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"### 🚶‍♂️ {away_team}")
        st.markdown(f"<p style='font-size:15px'>⚡ Tempo zápasu:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px'>{tempo_stats_away['rating']}</p>", unsafe_allow_html=True)

        st.markdown(f"<p style='font-size:15px'>🥾 Tvrdost zápasu:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px'>{tempo_stats_away['aggressiveness_rating']}</p>", unsafe_allow_html=True)

        st.markdown(f"<p style='font-size:15px'>📊 Dominance v zápase:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px'>{tempo_stats_away['imbalance_type']}</p>", unsafe_allow_html=True)



    # Nejpravděpodobnější výsledky
    # Výpočet
    top_scores = get_top_scorelines(matrix, top_n=5)
    home_probs, away_probs = get_goal_probabilities(matrix)

    # Dataframe – TOP skóre
    top_df = pd.DataFrame([
        {"Skóre": f"{a}:{b}", "Pravděpodobnost": f"{round(p*100, 1)} %"}
        for (a, b), p in top_scores
    ])

    # Dataframe – šance na góly
    goly = [1, 2, 3,4,5]
    goal_chances = pd.DataFrame({
        "Góly": goly,
        home_team: [f"{round(home_probs[i]*100, 1)} %" for i in goly],
        away_team: [f"{round(away_probs[i]*100, 1)} %" for i in goly],
    })

    # Zobrazení vedle sebe
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏅 Nejpravděpodobnější výsledky")
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🎯 Šance na počet vstřelených gólů")
        st.dataframe(goal_chances, use_container_width=True, hide_index=True)

else:
    # 🟣 HROMADNÝ REŽIM (nová funkce)

    st.title("📋 Hromadné predikce zápasů")

    # --- Přidání zápasu ---
    if st.sidebar.button("➕ Přidat zápas"):
        if home_team != away_team:
            st.session_state.match_list.append({
                "league_file": league_file,
                "league_name": league_name,
                "home_team": home_team,
                "away_team": away_team
            })
        else:
            st.warning("⚠️ Vyber různé týmy!")

    # --- Vymazání všech zápasů ---
    if st.sidebar.button("🗑️ Vymazat všechny zápasy"):
        st.session_state.match_list.clear()

    if st.session_state.match_list:
        export_data = []

        for idx, match in enumerate(st.session_state.match_list):
            with st.container():
                st.markdown("---")
                st.subheader(f"🔮 {match['home_team']} vs {match['away_team']} {match['league_name']}")

                try:
                    df_match = load_data(match["league_file"])
                    validate_dataset(df_match)
                    elo_dict = calculate_elo_ratings(df_match)
                    home_exp, away_exp = expected_goals_weighted_by_elo(df_match, match["home_team"], match["away_team"], elo_dict)
                    matrix = poisson_prediction(home_exp, away_exp)
                    outcomes = match_outcomes_prob(matrix)
                    over_under = over_under_prob(matrix)
                    btts = btts_prob(matrix)
                    xpoints = calculate_expected_points(outcomes)

                    # ⚽ Základní predikce: xG + BTTS + Over 2.5
                    cols = st.columns(3)
                    cols[0].metric("⚽ Očekávané góly", f"{home_exp:.1f} - {away_exp:.1f}")
                    cols[1].metric("🔵 BTTS %", f"{btts['BTTS Yes']}%")
                    cols[2].metric("📈 Over 2.5 %", f"{over_under['Over 2.5']}%")

                    # 🧠 Pravděpodobnosti výsledků
                    st.markdown("#### 🧠 Pravděpodobnosti výsledků")
                    result_cols = st.columns(3)
                    result_cols[0].metric("🏠 Výhra domácích", f"{outcomes['Home Win']}%", f"{prob_to_odds(outcomes['Home Win'])}")
                    result_cols[1].metric("🤝 Remíza", f"{outcomes['Draw']}%", f"{prob_to_odds(outcomes['Draw'])}")
                    result_cols[2].metric("🚶‍♂️ Výhra hostů", f"{outcomes['Away Win']}%", f"{prob_to_odds(outcomes['Away Win'])}")

                    # 🏅 Nejpravděpodobnější skóre
                    top_scores = get_top_scorelines(matrix, top_n=1)
                    if top_scores:
                        top_score, top_prob = top_scores[0]
                        st.markdown(f"#### 🏅 Nejpravděpodobnější skóre: **{top_score[0]}:{top_score[1]}**")


                    # --- Možnost odebrat tento zápas ---
                    if st.button(f"🗑️ Smazat zápas {match['home_team']} vs {match['away_team']}", key=f"del_{idx}"):
                        st.session_state.match_list.pop(idx)
                        st.st.rerun()
                        
                    # --- Přidání do exportu ---
                    top_scores = get_top_scorelines(matrix, top_n=1)
                    top_score = f"{top_scores[0][0][0]}:{top_scores[0][0][1]}"

                    export_data.append({
                        "League": match["league_name"],
                        "Home": match["home_team"],
                        "Away": match["away_team"],
                        "Home ExpG": round(home_exp, 2),
                        "Away ExpG": round(away_exp, 2),
                        "BTTS %": btts['BTTS Yes'],
                        "Over 2.5 %": over_under['Over 2.5'],
                        "Home Win %": outcomes["Home Win"],
                        "Draw %": outcomes["Draw"],
                        "Away Win %": outcomes["Away Win"],
                        "Top Score": top_score,
                    })

                except Exception as e:
                    st.error(f"⚠️ Chyba při predikci: {str(e)}")

        # --- Export CSV ---
        if export_data:
            df_export = pd.DataFrame(export_data)
            # csv = df_export.to_csv(index=False).encode('utf-8')
            # st.download_button(
            #     label="📥 Stáhnout predikce jako CSV",
            #     data=csv,
            #     file_name="multi_predictions.csv",
            #     mime="text/csv"
            # )
            
            # Aktuální datum a počet zápasů
            today = date.today().strftime("%Y-%m-%d")
            num_matches = len(export_data)
            file_suffix = f"{today}_{num_matches:02d}"
            file_name = f"multi_predictions_{file_suffix}.xlsx"

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Predictions')

            st.download_button(
                label="📥 Stáhnout predikce jako Excel",
                data=output.getvalue(),
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.info("👈 Přidej zápasy přes tlačítko ➕ v sidebaru.")




