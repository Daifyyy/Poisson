import time
import streamlit as st
from sections.overview_section import render_league_overview
from sections.match_prediction_section import render_single_match_prediction
from sections.multi_prediction_section import render_multi_match_predictions
from utils.poisson_utils import load_data, detect_current_season, calculate_team_strengths, calculate_gii_zscore,calculate_elo_ratings, get_team_average_gii
from utils.frontend_utils import validate_dataset
from utils.update_data import update_all_leagues

st.set_page_config(page_title="⚽ Poisson Predictor", layout="wide")

# Ligové soubory
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

# --- Sidebar: Správa dat ---
with st.sidebar.expander("🔧 Správa dat"):
    if st.button("🔄 Aktualizovat data z webu"):
        with st.spinner("Stahuji a porovnávám data..."):
            logs = update_all_leagues()
            if "reload_flag" in st.session_state:
                del st.session_state["reload_flag"]
            st.session_state.reload_flag = True

        placeholder = st.sidebar.empty()  # 💎 sidebar placeholder
        for log in logs:
            if "✅" in log:
                placeholder.success(log)
            else:
                placeholder.info(str(log))
            time.sleep(3)
            placeholder.empty()


# --- Sidebar: Výběr ligy ---
league_name = st.sidebar.selectbox("🌍 Vyber ligu", list(league_files.keys()))
league_file = league_files[league_name]

# --- Načtení a příprava dat ---
@st.cache_data(show_spinner=False)
def load_and_prepare(file_path):
    df = load_data(file_path)
    validate_dataset(df)
    season_df, _ = detect_current_season(df)
    team_strengths, _, _ = calculate_team_strengths(df)
    season_df = calculate_gii_zscore(season_df)
    gii_dict = get_team_average_gii(season_df)
    elo_dict = calculate_elo_ratings(df)
    return df, season_df, gii_dict, elo_dict


# --- Načtení s podmínkou opětovného načtení po aktualizaci ---
if st.session_state.get("reload_flag"):
    st.cache_data.clear()
    del st.session_state["reload_flag"]

df, season_df, gii_dict, elo_dict = load_and_prepare(league_file)

if "match_list" not in st.session_state:
    st.session_state.match_list = []

# Výběr týmů
teams_in_season = sorted(set(season_df["HomeTeam"].unique()) | set(season_df["AwayTeam"].unique()))
home_team = st.sidebar.selectbox("Domácí tým", teams_in_season)
away_team = st.sidebar.selectbox("Hostující tým", teams_in_season)
multi_prediction_mode = st.sidebar.checkbox("📝 Hromadné predikce")

if home_team == away_team:
    render_league_overview(season_df, league_name, gii_dict)
elif not multi_prediction_mode:
    render_single_match_prediction(df, season_df, home_team, away_team, league_name, gii_dict, elo_dict)
else:
    render_multi_match_predictions(st.session_state, home_team, away_team, league_name, league_file, league_files)
