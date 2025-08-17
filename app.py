import time
import sys
import os
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
from sections.my_bets_section import render_my_bets_section as render_my_bets
from sections.cross_league_section import render_cross_league_ratings

import urllib.parse

# Import only the required functions from the ``poisson_utils`` package
# to avoid importing heavy optional dependencies.  The package's ``__init__``
# pulls in many modules (such as API clients) which might not be available in
# minimal environments like the execution sandbox used for testing.  Importing
# directly from submodules keeps the import surface small and prevents an
# ``ImportError`` when optional requirements are missing.
from utils.poisson_utils.data import (
    load_data,
    detect_current_season,
    load_cup_matches,
    load_cup_team_stats,
)
from utils.poisson_utils.match_style import calculate_gii_zscore, get_team_average_gii
from utils.poisson_utils.elo import calculate_elo_ratings
from utils.poisson_utils.cross_league import (
    calculate_cross_league_team_index,
    build_league_quality_table,
)
from utils.frontend_utils import validate_dataset
from utils.update_data import update_all_leagues
from utils.bet_db import init_db

# Initialize bets database on startup
init_db()

# --- Základní nastavení ---
# Layout can be switched via env variable STREAMLIT_LAYOUT; defaults to "wide"
layout = os.getenv("STREAMLIT_LAYOUT", "wide")
if layout not in {"wide", "centered"}:
    layout = "wide"
st.set_page_config(
    page_title="⚽ Poisson Predictor",
    page_icon="⚽",
    layout=layout,
    initial_sidebar_state="expanded",
)
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

# --- Sidebar: Navigation ---
if st.sidebar.button("🏠 Home"):
    st.session_state.clear()
    st.query_params.clear()
    st.rerun()

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


@st.cache_data(show_spinner=False)
def compute_cross_league_index(files: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cross-league team index for all leagues in ``files``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Team ratings and league quality tables.
    """

    team_frames = []
    match_frames = []
    team_league_map = {}

    for _, path in files.items():
        league_df = load_data(path)

        match_frames.append(
            league_df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
        )

        # build mapping from team to domestic league
        league_code = league_df["Div"].iloc[0]
        teams = pd.concat([league_df["HomeTeam"], league_df["AwayTeam"]]).unique()
        for team in teams:
            team_league_map[team] = league_code

        # Aggregate totals for each team
        home = league_df[
            [
                "Div",
                "HomeTeam",
                "FTHG",
                "FTAG",
                "HS",
                "HST",
                "AS",
                "AST",
            ]
        ].rename(
            columns={
                "Div": "league",
                "HomeTeam": "team",
                "FTHG": "goals_for",
                "FTAG": "goals_against",
                "HS": "shots_for",
                "HST": "sot_for",
                "AS": "shots_against",
                "AST": "sot_against",
            }
        )

        away = league_df[
            [
                "Div",
                "AwayTeam",
                "FTAG",
                "FTHG",
                "AS",
                "AST",
                "HS",
                "HST",
            ]
        ].rename(
            columns={
                "Div": "league",
                "AwayTeam": "team",
                "FTAG": "goals_for",
                "FTHG": "goals_against",
                "AS": "shots_for",
                "AST": "sot_for",
                "HS": "shots_against",
                "HST": "sot_against",
            }
        )

        combined = pd.concat([home, away], ignore_index=True)
        agg = (
            combined.groupby(["league", "team"])
            .agg(
                matches=("goals_for", "size"),
                goals_for=("goals_for", "sum"),
                goals_against=("goals_against", "sum"),
                shots_for=("shots_for", "sum"),
                shots_against=("shots_against", "sum"),
                sot_for=("sot_for", "sum"),
                sot_against=("sot_against", "sum"),
            )
            .reset_index()
        )

        agg["xg_for"] = 0.1 * agg["shots_for"] + 0.3 * agg["sot_for"]
        agg["xg_against"] = 0.1 * agg["shots_against"] + 0.3 * agg["sot_against"]
        agg = agg.drop(columns=["sot_for", "sot_against"])

        team_frames.append(agg)

    # append cup matches if available
    data_dir = Path(next(iter(files.values()))).parent
    cup_df = load_cup_matches(team_league_map, data_dir)
    if not cup_df.empty:
        match_frames.append(cup_df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
        cup_stats = load_cup_team_stats(team_league_map, data_dir, cup_df)
        team_frames.append(cup_stats)

    teams_df = pd.concat(team_frames, ignore_index=True)
    teams_df = (
        teams_df.groupby(["league", "team"], as_index=False)
        .sum(numeric_only=True)
    )
    matches_df = pd.concat(match_frames, ignore_index=True)
    ratings_df = pd.read_csv(ROOT / "data" / "league_penalty_coefficients.csv")
    team_df = calculate_cross_league_team_index(teams_df, ratings_df, matches_df)
    if "penalty_coef" in ratings_df.columns:
        league_df = ratings_df.rename(columns={"penalty_coef": "league_penalty_coef"})
    else:
        league_df = build_league_quality_table(ratings_df)
    return team_df, league_df

# Re-load po aktualizaci dat
if st.session_state.get("reload_flag"):
    st.cache_data.clear()
    del st.session_state["reload_flag"]

df, season_df, gii_dict, elo_dict = load_and_prepare(league_file)
# Zachováme kompletní dataset pro historické statistiky (H2H apod.)
full_df = df.copy()

# Cross-league ratings for all teams
cross_league_df, league_quality_df = compute_cross_league_index(league_files)

# --- Date range filtr ---
overall_start = df["Date"].min().date()
overall_end = df["Date"].max().date()
season_start = season_df["Date"].min().date()
season_end = season_df["Date"].max().date()
start_date = st.sidebar.date_input(
    "📅 Začátek", season_start, min_value=overall_start, max_value=overall_end
)
end_date = st.sidebar.date_input(
    "📅 Konec", season_end, min_value=overall_start, max_value=overall_end
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

# --- Navigation ---
navigation = st.sidebar.radio(
    "Navigate",
    (
        "League overview",
        "Match prediction",
        "Multi predictions",
        "Cross-league ratings",
        "My Bets",
    ),
)

# --- Query params ---
query_params = st.query_params

# Clear selected team when switching navigation to avoid stale selection
if "last_navigation" not in st.session_state:
    st.session_state["last_navigation"] = navigation
elif st.session_state["last_navigation"] != navigation:
    st.session_state["last_navigation"] = navigation
    if "selected_team" in query_params:
        del query_params["selected_team"]

# --- Výběr týmů ---
teams_in_season = sorted(
    set(season_df["HomeTeam"].unique()) | set(season_df["AwayTeam"].unique())
)
home_team = away_team = None
if navigation in ("Match prediction", "Multi predictions"):
    home_team = st.sidebar.selectbox("Domácí tým", teams_in_season)
    away_team = st.sidebar.selectbox("Hostující tým", teams_in_season)

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
if navigation == "My Bets":
    render_my_bets()

elif (
    navigation == "Match prediction"
    and home_team
    and away_team
    and home_team != away_team
):
    render_single_match_prediction(
        df,
        season_df,
        full_df,
        home_team,
        away_team,
        league_name,
        gii_dict,
        elo_dict,
    )

elif navigation == "Multi predictions":
    render_multi_match_predictions(
        st.session_state,
        home_team,
        away_team,
        league_name,
        league_file,
        league_files,
    )

elif navigation == "Cross-league ratings":
    render_cross_league_ratings(cross_league_df, league_quality_df)

elif selected_team:
    render_team_detail(df, season_df, selected_team, league_name, gii_dict)
    if st.button("🔙 Zpět na ligový přehled"):
        st.query_params.clear()
        st.query_params["selected_league"] = league_name
        st.rerun()

else:
    render_league_overview(season_df, league_name, gii_dict, elo_dict)
