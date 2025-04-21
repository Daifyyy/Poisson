
import streamlit as st
import pandas as pd
from utils.poisson_utils import (
    load_data, validate_dataset, calculate_team_strengths,
    expected_goals_weighted_final, expected_team_stats_weighted,
    poisson_prediction, match_outcomes_prob, over_under_prob,
    btts_prob, prob_to_odds, calculate_pseudo_xg,
    analyze_opponent_strength, calculate_expected_points,
    generate_score_heatmap, get_top_scorelines, plot_top_scorelines
)

st.set_page_config(page_title="⚽ Poisson Predictor", layout="wide")
st.title("⚽ Poisson Match Predictor")

# Výběr ligy a načtení dat
league_files = {
    "E0 (Premier League)": "data/E0_combined_full.csv",
    "SP1 (La Liga)": "data/SP1_combined_full.csv"
}
league_name = st.sidebar.selectbox("🌍 Vyber ligu", list(league_files.keys()))
league_file = league_files[league_name]

df = load_data(league_file)
validate_dataset(df)
team_strengths, _, _ = calculate_team_strengths(df)

# Výběr týmů
st.sidebar.header("🏟️ Zápas")
home_team = st.sidebar.selectbox("Domácí tým", team_strengths["Team"].sort_values())
away_team = st.sidebar.selectbox("Hostující tým", team_strengths["Team"].sort_values())

if home_team == away_team:
    st.header("📊 Přehled ligy")
    avg_stats = df.groupby("HomeTeam").agg({
        "FTHG": "mean",
        "FTAG": "mean",
        "HS": "mean",
        "HST": "mean",
        "HC": "mean",
        "HY": "mean"
    }).rename(columns={
        "FTHG": "Góly doma",
        "FTAG": "Góly venku",
        "HS": "Střely",
        "HST": "Na branku",
        "HC": "Rohy",
        "HY": "Žluté"
    })
    st.dataframe(avg_stats.round(2))
    st.stop()

st.header(f"🔮 {home_team} vs {away_team}")

# Výpočty
home_exp, away_exp = expected_goals_weighted_final(df, home_team, away_team)
matrix = poisson_prediction(home_exp, away_exp)
outcomes = match_outcomes_prob(matrix)
over_under = over_under_prob(matrix)
btts = btts_prob(matrix)
xpoints = calculate_expected_points(outcomes)

xg_home = calculate_pseudo_xg(df, home_team)
xg_away = calculate_pseudo_xg(df, away_team)

strength_home = analyze_opponent_strength(df, home_team, is_home=True)
strength_away = analyze_opponent_strength(df, away_team, is_home=False)

# 🧮 Skóre
st.markdown("## ⚽ Očekávané skóre")
st.markdown(f"### `{home_team}` **{round(home_exp, 2)}** : **{round(away_exp, 2)}** `{away_team}`")

# 🔢 Statistiky v řádku
st.markdown("## 📊 Klíčové metriky")
cols = st.columns(4)
cols[0].metric("xG sezóna", f"{xg_home['avg_xG']} vs {xg_away['avg_xG']}")
cols[1].metric("xG (posledních 5)", f"{xg_home['xG_last5']} vs {xg_away['xG_last5']}")
cols[2].metric("Oček. body (xP)", f"{xpoints['Home xP']} vs {xpoints['Away xP']}")
cols[3].metric("BTTS / Over 2.5", f"{btts['BTTS Yes']}% / {over_under['Over 2.5']}%")
cols[3].caption(f"Kurzy: {prob_to_odds(btts['BTTS Yes'])} / {prob_to_odds(over_under['Over 2.5'])}")

# 🧠 Pravděpodobnosti výsledků
st.markdown("## 🧠 Pravděpodobnosti výsledků")
cols2 = st.columns(3)
cols2[0].metric("🏠 Výhra domácích", f"{outcomes['Home Win']}%", f"{prob_to_odds(outcomes['Home Win'])}")
cols2[1].metric("🤝 Remíza", f"{outcomes['Draw']}%", f"{prob_to_odds(outcomes['Draw'])}")
cols2[2].metric("🚶‍♂️ Výhra hostů", f"{outcomes['Away Win']}%", f"{prob_to_odds(outcomes['Away Win'])}")

# 📌 Týmové statistiky
st.markdown("## 📌 Týmové statistiky (vážený průměr)")
stat_map = {
    'Střely': ('HS', 'AS'),
    'Střely na branku': ('HST', 'AST'),
    'Rohy': ('HC', 'AC'),
    'Žluté karty': ('HY', 'AY')
}
extra_stats = expected_team_stats_weighted(df, home_team, away_team, stat_map)
for stat, values in extra_stats.items():
    st.markdown(f"- **{stat}**: `{home_team}` {values['Home']} – {values['Away']} `{away_team}`")

# 📊 Výkon vůči soupeřům
st.markdown("## ⚖️ Forma proti soupeřům")
cols3 = st.columns(2)
with cols3[0]:
    st.metric(f"{home_team} vs silní", strength_home['conversion_vs_strong'])
    st.metric(f"{home_team} vs slabí", strength_home['conversion_vs_weak'])
    st.caption(f"Celková konverze: {strength_home['overall_conversion']}")
with cols3[1]:
    st.metric(f"{away_team} vs silní", strength_away['conversion_vs_strong'])
    st.metric(f"{away_team} vs slabí", strength_away['conversion_vs_weak'])
    st.caption(f"Celková konverze: {strength_away['overall_conversion']}")

# 🔥 Heatmapa
st.markdown("## 📊 Heatmapa skóre")
st.pyplot(generate_score_heatmap(matrix, home_team, away_team))

# 🥇 Top skóre
st.markdown("## 🥇 Nejpravděpodobnější výsledky")
score_probs = get_top_scorelines(matrix)
st.pyplot(plot_top_scorelines(score_probs, home_team, away_team))
