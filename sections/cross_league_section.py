import streamlit as st
import pandas as pd


def render_cross_league_ratings(df: pd.DataFrame) -> None:
    """Display cross-league team ratings with optional filtering."""

    st.header("🌍 Cross-League Team Ratings")

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

    display_cols = {
        "league": "League",
        "team": "Team",
        "team_index": "Team Strength",
        "team_elo_rel": "ELO Adj",
        "xg_diff_norm": "xG Diff Adj",
        "sos": "SOS Adj",
    }
    table = (
        filtered.sort_values("team_index", ascending=False)[display_cols.keys()]
        .reset_index(drop=True)
        .rename(columns=display_cols)
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.caption(
        """
**Legend**
- **Team Strength** – overall team rating relative to global average.
- **ELO Adj** – team's ELO rating relative to its league average (1.0 is league mean).
- **xG Diff Adj** – expected goals differential normalised within its league.
- **SOS Adj** – strength of schedule z-score based on opponent quality.
        """
    )
