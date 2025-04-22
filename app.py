import streamlit as st
import pandas as pd
from utils.poisson_utils import (
    load_data, validate_dataset, calculate_team_strengths,
    expected_goals_weighted_final, expected_team_stats_weighted,
    poisson_prediction, match_outcomes_prob, over_under_prob,
    btts_prob, prob_to_odds,calculate_pseudo_xg,
    analyze_opponent_strength, calculate_expected_points,
    generate_score_heatmap, get_top_scorelines, plot_top_scorelines,
    calculate_elo_ratings, calculate_recent_form, detect_current_season,
    calculate_team_pseudo_xg,calculate_expected_and_actual_points,
)

st.set_page_config(page_title="⚽ Poisson Predictor", layout="wide")
st.title("⚽ Poisson Match Predictor")

# Výběr ligy a načtení dat
league_files = {
    "E0 (Premier League)": "data/E0_combined_full_updated.csv",
    "E1 (Championship)": "data/E1_combined_full_updated.csv",  
    "SP1 (La Liga)": "data/SP1_combined_full.csv",
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
print(season_df)
print(season_start)
# Výběr týmů
st.sidebar.header("🏟️ Zápas")
home_team = st.sidebar.selectbox("Domácí tým", team_strengths["Team"].sort_values())
away_team = st.sidebar.selectbox("Hostující tým", team_strengths["Team"].sort_values())

if home_team == away_team:
    st.header(f"🏆 {league_name} – Přehled sezóny")

    num_matches = len(season_df)
    avg_goals = round((season_df['FTHG'] + season_df['FTAG']).mean(), 2)
    season_df['BTTS'] = season_df.apply(lambda row: int(row['FTHG'] > 0 and row['FTAG'] > 0), axis=1)
    btts_pct = round(100 * season_df['BTTS'].mean(), 1)
    over_25 = round(100 * season_df[(season_df['FTHG'] + season_df['FTAG']) > 2.5].shape[0] / num_matches, 1)

    st.markdown(f"📅 Zápasů: {num_matches} ⚽ Průměr gólů: {avg_goals} 🥅 BTTS: {btts_pct}% 📈 Over 2.5: {over_25}%")

    elo_dict = calculate_elo_ratings(df)
    form_dict = calculate_recent_form(df, days=30)
    points_data = calculate_expected_and_actual_points(season_df)

    team_stats = season_df.groupby("HomeTeam").agg({
        "FTHG": "mean", "FTAG": "mean", "HS": "mean", "HST": "mean", "HC": "mean", "HY": "mean"
    }).rename(columns={
        "FTHG": "Góly doma", "FTAG": "Góly venku", "HS": "Střely", "HST": "Na branku", "HC": "Rohy", "HY": "Žluté"
    })

    over25 = season_df.groupby("HomeTeam").apply(lambda x: (x['FTHG'] + x['FTAG'] > 2.5).mean() * 100).round(0)
    btts = season_df.groupby("HomeTeam")["BTTS"].mean().mul(100).round(0)

    home_away_stats = season_df.groupby("HomeTeam").agg({
    "FTHG": "mean", "FTAG": "mean",
    "HS": "mean", "HST": "mean"
    }).rename(columns={
        "FTHG": "Góly doma", "FTAG": "Góly venku", "HS": "Střely doma", "HST": "Na branku doma"
    })

    away_additional = season_df.groupby("AwayTeam").agg({
        "FTHG": "mean", "FTAG": "mean",
        "HS": "mean", "HST": "mean"
    }).rename(columns={
        "FTHG": "Góly obdržené doma",
        "FTAG": "Góly venku (away)",  
        "HS": "Střely venku",
        "HST": "Na branku venku"
    })

    full_stats = home_away_stats.join(away_additional, how='outer')
    xg_stats = calculate_team_pseudo_xg(season_df)
    summary_table = pd.DataFrame({
            "Tým": team_stats.index,
            "Elo": team_stats.index.map(lambda t: elo_dict.get(t, 1500)),
            "Body": team_stats.index.map(lambda t: points_data.get(t, {}).get("points", 0)),
            #"xP": team_stats.index.map(lambda t: points_data.get(t, {}).get("expected_points", 0)),
            "Form": team_stats.index.map(lambda t: "🔥🔥🔥" if form_dict.get(t, 0) >= 2.4 else "🔥🔥" if form_dict.get(t, 0) >= 2.0 else "🔥" if form_dict.get(t, 0) >= 1.5 else "❄️❄️" if form_dict.get(t, 0) < 1.0 else "❄️"),
            "Góly/zápas": ((team_stats["Góly doma"] + team_stats["Góly venku"]) / 2).round(2),
            "xG/zápas": team_stats.index.map(lambda t: round(xg_stats.get(t, {}).get("avg_xG", 0), 2)),
            "xG/gól": team_stats.index.map(lambda t: round(xg_stats.get(t, {}).get("xG_per_goal", 0), 2)),
            "Over 2.5 %": team_stats.index.map(over25).astype(str) + "%",
            "BTTS %": team_stats.index.map(btts).astype(str) + "%"
        }).sort_values("Elo", ascending=False).reset_index(drop=True)

    st.dataframe(summary_table, hide_index=True)

    st.markdown("### 🌟 3. Top 5 týmy v různých kategoriích")
    cols = st.columns(4)

    with cols[0]:
        st.markdown("🔮 **Nejvíc gólů**")
        st.dataframe(summary_table.sort_values("Góly/zápas", ascending=False).head(5)[["Tým", "Góly/zápas"]], hide_index=True)

    with cols[1]:
        st.markdown("🛡️ **Nejlepší obrana (góly venku)**")
        st.dataframe(team_stats[["Góly venku"]].sort_values("Góly venku").head(5).reset_index().rename(columns={"HomeTeam": "Tým"}), hide_index=True)

    with cols[2]:
        st.markdown("🎯 **Nejlepší konverze (xG/gól)**")
        st.dataframe(summary_table.sort_values("xG/gól", ascending=False).head(5)[["Tým", "xG/gól"]], hide_index=True)

    with cols[3]:
        st.markdown("📈 **Nejlepší forma (body/zápas)**")
        st.dataframe(summary_table.sort_values("Form", ascending=False).head(5)[["Tým", "Form"]], hide_index=True)


    st.stop()

st.header(f"🔮 {home_team} vs {away_team}")

try:
    home_exp, away_exp = expected_goals_weighted_final(df, home_team, away_team)
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
st.markdown("## ⚽ Očekávané skóre")
st.markdown(f"### `{home_team}` **{round(home_exp, 2)}** : **{round(away_exp, 2)}** `{away_team}`")

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

# 📊 Výkon vůči soupeřům (detailní přehled)
# 📊 Výkon vůči soupeřům (detailní přehled)
st.markdown("## ⚖️ Výkon proti typům soupeřů")
perf_home = analyze_opponent_strength(df, home_team, is_home=True)
perf_away = analyze_opponent_strength(df, away_team, is_home=False)

st.markdown(f"### 🏠 Výkon domácího týmu – {home_team}")
home_cols = st.columns(3)
with st.container():
    for i, cat in enumerate(['vs_strong', 'vs_average', 'vs_weak']):
        with home_cols[i]:
            st.metric("Zápasy", perf_home[cat]['matches'])
            st.metric("Góly", perf_home[cat]['goals'])
            st.metric("Konverze", f"{perf_home[cat]['con_rate']*100:.1f}%")
            st.metric("Body/zápas", perf_home[cat]['xP'])
            st.caption(["💪 Silní", "⚖️ Průměrní", "🪶 Slabí"][i])

st.markdown(f"### 🚶‍♂️ Výkon hostujícího týmu – {away_team}")
away_cols = st.columns(3)
with st.container():
    for i, cat in enumerate(['vs_strong', 'vs_average', 'vs_weak']):
        with away_cols[i]:
            st.metric("Zápasy", perf_away[cat]['matches'])
            st.metric("Góly", perf_away[cat]['goals'])
            st.metric("Konverze", f"{perf_away[cat]['con_rate']*100:.1f}%")
            st.metric("Body/zápas", perf_away[cat]['xP'])
            st.caption(["💪 Silní", "⚖️ Průměrní", "🪶 Slabí"][i])







# 🔥 Heatmapa
st.markdown("## 📊 Heatmapa skóre")
st.pyplot(generate_score_heatmap(matrix, home_team, away_team))


