import streamlit as st
import pandas as pd
from pathlib import Path

from utils.poisson_utils.cup_predictions import predict_cup_match
from utils.poisson_utils.data import prepare_df

CUP_FILES = {
    "Champions League": "CL_combined_full.csv",
    "Europa League": "EL_combined_full.csv",
    "Conference League": "ECL_combined_full.csv",
}


@st.cache_data
def load_upcoming_cup_fixtures(data_dir: str | Path = "data") -> pd.DataFrame:
    """Return upcoming fixtures from predefined UEFA cup CSV files.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the cup CSV files. Defaults to ``"data"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Date``, ``Competition``, ``HomeTeam`` and
        ``AwayTeam`` for matches where scores are missing.
    """

    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []
    for competition, filename in CUP_FILES.items():
        path = data_dir / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df = prepare_df(df)
        if {"FTHG", "FTAG"}.issubset(df.columns):
            df = df[df["FTHG"].isna() & df["FTAG"].isna()]
        else:
            df = df.iloc[0:0]
        if df.empty:
            continue
        df = df[["Date", "HomeTeam", "AwayTeam"]].copy()
        df["Competition"] = competition
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", "Competition", "HomeTeam", "AwayTeam"])

    return pd.concat(frames, ignore_index=True).sort_values("Date")


def render_uefa_cup_predictions(cross_league_df: pd.DataFrame, data_dir: str | Path = "data") -> None:
    """Render table of predicted outcomes for upcoming UEFA cup fixtures."""

    st.header("🏆 UEFA Cup Predictions")
    fixtures = load_upcoming_cup_fixtures(data_dir)
    if fixtures.empty:
        st.info("No upcoming fixtures found for the Champions League, Europa League or Conference League.")
        return

    rows: list[dict] = []
    for row in fixtures.itertuples(index=False):
        try:
            preds = predict_cup_match(row.HomeTeam, row.AwayTeam, cross_league_df)
        except KeyError:
            continue
        rows.append(
            {
                "Date": row.Date.date(),
                "Competition": row.Competition,
                "Home": row.HomeTeam,
                "Away": row.AwayTeam,
                "Home xG": preds["home_exp_goals"],
                "Away xG": preds["away_exp_goals"],
                "Home Win %": preds["home_win_pct"],
                "Draw %": preds["draw_pct"],
                "Away Win %": preds["away_win_pct"],
            }
        )

    if not rows:
        st.info("No predictions available for the loaded fixtures.")
        return

    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)
