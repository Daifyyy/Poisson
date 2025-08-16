import streamlit as st
import pandas as pd

def render_cross_league_index(df: pd.DataFrame) -> None:
    """Display cross-league team index with optional filtering."""

    st.header("🌍 Cross-League Team Index")

    if df.empty:
        st.info("No team data available.")
        return

    leagues = sorted(df["league"].unique())
    selected_leagues = st.multiselect("Leagues", leagues, default=leagues)
    filtered = df[df["league"].isin(selected_leagues)]

    teams = sorted(filtered["team"].unique())
    selected_teams = st.multiselect("Teams", teams)
    if selected_teams:
        filtered = filtered[filtered["team"].isin(selected_teams)]

    display_cols = [
        "league",
        "team",
        "team_index",
        "xg_vs_world",
        "off_rating",
        "def_rating",
    ]
    st.dataframe(
        filtered.sort_values("team_index", ascending=False)[display_cols].reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )
