import streamlit as st
import pandas as pd
from pathlib import Path

from utils.poisson_utils.cup_predictions import predict_cup_match
from utils.poisson_utils.data import prepare_df
from utils.poisson_utils.elo import calculate_elo_ratings

# Single combined CSV with all UEFA cup matches.  The ``Competition`` column
# stores short codes (CL, EL, ECL) which we map to human readable names below.
CUP_FILE = "all_cups_combined.csv"
COMPETITION_NAMES = {
    "CL": "Champions League",
    "EL": "Europa League",
    "ECL": "Conference League",
}


@st.cache_data
def load_upcoming_cup_fixtures(data_dir: str | Path = "data") -> pd.DataFrame:
    """Return upcoming fixtures from the combined UEFA cup CSV file.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``all_cups_combined.csv``. Defaults to ``"data"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Date``, ``Competition``, ``HomeTeam`` and
        ``AwayTeam`` for matches where scores are missing.
    """

    data_dir = Path(data_dir)
    path = data_dir / CUP_FILE
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Competition", "HomeTeam", "AwayTeam"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["Date", "Competition", "HomeTeam", "AwayTeam"])

    df = prepare_df(df)
    if {"FTHG", "FTAG"}.issubset(df.columns):
        df = df[df["FTHG"].isna() & df["FTAG"].isna()]
    else:
        return pd.DataFrame(columns=["Date", "Competition", "HomeTeam", "AwayTeam"])

    if df.empty:
        return pd.DataFrame(columns=["Date", "Competition", "HomeTeam", "AwayTeam"])

    df = df[["Date", "HomeTeam", "AwayTeam", "Competition"]].copy()
    df["Competition"] = df["Competition"].map(COMPETITION_NAMES).fillna(df["Competition"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_cup_elo_tables(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """Return ELO rating tables for each competition in the combined CSV."""

    data_dir = Path(data_dir)
    path = data_dir / CUP_FILE
    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    df = prepare_df(df)
    if not {"FTHG", "FTAG", "Competition"}.issubset(df.columns):
        return {}

    df = df.dropna(subset=["FTHG", "FTAG"])
    if df.empty:
        return {}

    tables: dict[str, pd.DataFrame] = {}
    for comp_code, comp_df in df.groupby("Competition"):
        elo_dict = calculate_elo_ratings(comp_df)
        elo_table = (
            pd.DataFrame({"Team": elo_dict.keys(), "ELO": elo_dict.values()})
            .sort_values("ELO", ascending=False)
            .reset_index(drop=True)
        )
        comp_name = COMPETITION_NAMES.get(comp_code, comp_code)
        tables[comp_name] = elo_table

    return tables


def render_uefa_cup_predictions(cross_league_df: pd.DataFrame, data_dir: str | Path = "data") -> None:
    """Render table of predicted outcomes for upcoming UEFA cup fixtures."""

    st.header("🏆 UEFA Cup Predictions")
    elo_tables = load_cup_elo_tables(data_dir)
    for competition, table in elo_tables.items():
        st.subheader(f"{competition} ELO Ratings")
        st.dataframe(table, hide_index=True, use_container_width=True)

    fixtures = load_upcoming_cup_fixtures(data_dir)
    if fixtures.empty:
        st.info(
            "No upcoming fixtures found for the Champions League, Europa League or Conference League."
        )
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
