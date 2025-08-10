import time
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from utils.responsive import init_responsive_layout

# Umožní import lokálních balíčků i při spuštění z nadřazeného adresáře
# (např. `streamlit run poisson/app.py`)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sections.overview_section import render_league_overview
from sections.match_prediction_section import render_single_match_prediction
from sections.multi_prediction_section import render_multi_match_predictions
from sections.team_detail_section import render_team_detail

import urllib.parse
from utils.poisson_utils import (
    load_data,
    detect_current_season,
    calculate_gii_zscore,
    calculate_elo_ratings,
    get_team_average_gii,
)
from utils.frontend_utils import validate_dataset
from utils.update_data import update_all_leagues

# --- Základní nastavení ---
st.set_page_config(page_title="⚽ Poisson Predictor", layout="wide")
init_responsive_layout()
pd.options.display.float_format = lambda x: f"{x:.1f}"

# --- Cesty k ligovým souborům ---
league_files = {
    "E0 (Premier League)": "data/E0_combined_full_updated.csv",
    "E1 (Championship)": "data/E1_combined_full_updated.csv",
    "SP1 (La Liga)": "data/SP1_combined_full_updated.csv",
    "B1 (Jupiler League)": "data/B1_combined_full_updated.csv",
    "D1 (Bundesliga)": "data/D1_combined_full_updated.csv",
    "D2 (2. Bundesliga)": "data/D2_combined_full_updated.csv",
    "I1 (Seria A)": "data/I1_combined_full_updated.csv",
    "F1 (Ligue 1)": "data/F1_combined_full_updated.csv",
    "N1 (Eredivisie)": "data/N1_combined_full_updated.csv",
    "P1 (Primeira Liga)": "data/P1_combined_full_updated.csv",
    "T1 (Super League)": "data/T1_combined_full_updated.csv",
}

# Měsíce typického startu sezony (fallback, když se nedetekuje pauza)
LEAGUE_START_MONTH = {
    "B1": 7,  # Jupiler League
    "D2": 7,  # 2. Bundesliga
    # ostatní implicitně 8
}

# --- Sidebar: Správa dat ---
with st.sidebar.expander("🔧 Správa dat"):
    if st.button("🔄 Aktualizovat data z webu"):
        with st.spinner("Stahuji a porovnávám data..."):
            logs = update_all_leagues()
            # signalizace pro re-load cache
            if "reload_flag" in st.session_state:
                del st.session_state["reload_flag"]
            st.session_state.reload_flag = True

        placeholder = st.sidebar.empty()
        for log in logs:
            if "✅" in log:
                placeholder.success(log)
            else:
                placeholder.info(str(log))
            time.sleep(3)
            placeholder.empty()

# --- Volba ligy (URL -> default) ---
selected_league_from_url = st.query_params.get("selected_league", None)
if isinstance(selected_league_from_url, list):
    selected_league_from_url = selected_league_from_url[0]

default_league_index = (
    list(league_files.keys()).index(selected_league_from_url)
    if selected_league_from_url in league_files
    else 0
)

league_name = st.sidebar.selectbox(
    "🌍 Vyber ligu", list(league_files.keys()), index=default_league_index
)
league_file = league_files[league_name]

# --- Načtení a příprava dat ---
@st.cache_data(show_spinner=False)
def load_and_prepare(file_path: str):
    df = load_data(file_path)
    validate_dataset(df)

    league_code = file_path.split("/")[-1].split("_")[0]
    start_month = LEAGUE_START_MONTH.get(league_code, 8)

    season_df, _ = detect_current_season(
        df, start_month=start_month, prepared=True
    )
    season_df = calculate_gii_zscore(season_df)

    gii_dict = get_team_average_gii(season_df)
    elo_dict = calculate_elo_ratings(df)

    return df, season_df, gii_dict, elo_dict

# Re-load po aktualizaci dat
if st.session_state.get("reload_flag"):
    st.cache_data.clear()
    del st.session_state["reload_flag"]

df, season_df, gii_dict, elo_dict = load_and_prepare(league_file)

# --- Date range filtr ---
start_default = season_df["Date"].min().date()
end_default = season_df["Date"].max().date()
start_date = st.sidebar.date_input(
    "📅 Začátek", start_default, min_value=start_default, max_value=end_default
)
end_date = st.sidebar.date_input(
    "📅 Konec", end_default, min_value=start_default, max_value=end_default
)

df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]
season_df = season_df[
    (season_df["Date"].dt.date >= start_date) & (season_df["Date"].dt.date <= end_date)
]

# využití předpočítaných hodnot místo kompletního přepočtu
filtered_teams = set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"]))
gii_dict = {team: gii_dict.get(team) for team in filtered_teams}
elo_dict = {team: elo_dict.get(team) for team in filtered_teams}

if "match_list" not in st.session_state:
    st.session_state.match_list = []

# --- Výběr týmů ---
teams_in_season = sorted(set(season_df["HomeTeam"].unique()) | set(season_df["AwayTeam"].unique()))
home_team = st.sidebar.selectbox("Domácí tým", teams_in_season)
away_team = st.sidebar.selectbox("Hostující tým", teams_in_season)
multi_prediction_mode = st.sidebar.checkbox("📝 Hromadné predikce")

# --- Query params ---
query_params = st.query_params
raw_team = query_params.get("selected_team", None)
if isinstance(raw_team, list):
    raw_team = raw_team[0]
selected_team = urllib.parse.unquote_plus(raw_team) if raw_team else None

# liga z query param (pro jistotu stejně jako výše)
selected_league_from_url = query_params.get("selected_league", None)
if isinstance(selected_league_from_url, list):
    selected_league_from_url = selected_league_from_url[0]

# --- Detekce změny ligy ---
if "last_selected_league" not in st.session_state:
    st.session_state["last_selected_league"] = league_name
elif st.session_state["last_selected_league"] != league_name:
    selected_team = None
    query_params.clear()
    query_params["selected_league"] = league_name
    st.session_state["last_selected_league"] = league_name
    st.rerun()

# === ROUTING ===
if selected_team:
    render_team_detail(df, season_df, selected_team, league_name, gii_dict)
    if st.button("🔙 Zpět na ligový přehled"):
        st.query_params.clear()
        st.query_params["selected_league"] = league_name
        st.rerun()

elif multi_prediction_mode:
    render_multi_match_predictions(
        st.session_state,
        home_team,
        away_team,
        league_name,
        league_file,
        league_files,
    )

elif home_team != away_team:
    render_single_match_prediction(
        df, season_df, home_team, away_team, league_name, gii_dict, elo_dict
    )

else:
    render_league_overview(season_df, league_name, gii_dict, elo_dict)
