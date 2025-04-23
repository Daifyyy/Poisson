import streamlit as st
import pandas as pd
import numpy as np
from utils.poisson_utils import (
    load_data, validate_dataset, calculate_team_strengths,
    
    poisson_prediction, match_outcomes_prob, over_under_prob,
    btts_prob, prob_to_odds,calculate_pseudo_xg,
    analyze_opponent_strength, calculate_expected_points,
    generate_score_heatmap, expected_goals_weighted_by_elo,
    calculate_elo_ratings, calculate_recent_form, detect_current_season,expected_team_stats_weighted_by_elo,
    calculate_team_pseudo_xg,calculate_expected_and_actual_points,merged_home_away_opponent_form
)

st.set_page_config(page_title="⚽ Poisson Predictor", layout="wide")
st.title("⚽ Poisson Match Predictor")

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
print(season_df)
print(season_start)
# Výběr týmů
st.sidebar.header("🏟️ Zápas")
teams_in_season = sorted(set(season_df["HomeTeam"].unique()) | set(season_df["AwayTeam"].unique()))
home_team = st.sidebar.selectbox("Domácí tým", teams_in_season)
away_team = st.sidebar.selectbox("Hostující tým", teams_in_season)


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
        "Konverze (%)": team_stats.index.map(lambda t: round(100 * (team_stats.loc[t, "Góly doma"] + team_stats.loc[t, "Góly venku"]) / (combined_sot.loc[t, "Celkem na branku"] + 0.1), 1)),        
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
st.markdown("## ⚽ Očekávané skóre")
st.markdown(f"### `{home_team}` **{round(home_exp, 1)}** : **{round(away_exp, 1)}** `{away_team}`")

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

def display_merged_table(data, team_name):
    st.markdown(f"### {team_name}")
    df_disp = pd.DataFrame(data).T  # index = ['💪 Silní', '⚖️ Průměrní', '🪶 Slabí']
    df_disp = df_disp[["Zápasy", "Góly", "Obdržené", "Střely", "Na branku", "xG", "Body/zápas"]]
    st.dataframe(df_disp)

merged_home = merged_home_away_opponent_form(df, home_team)
merged_away = merged_home_away_opponent_form(df, away_team)

display_merged_table(merged_home, home_team)
display_merged_table(merged_away, away_team)

# 🔥 Heatmapa
st.markdown("## 📊 Heatmapa skóre")
st.pyplot(generate_score_heatmap(matrix, home_team, away_team))


