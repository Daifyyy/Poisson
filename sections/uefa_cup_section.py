import os
import streamlit as st
import pandas as pd
import numpy as np
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
def load_upcoming_cup_fixtures(
    data_dir: str | Path = "data", mtime: float | None = None
) -> pd.DataFrame:
    """Return upcoming fixtures from the combined UEFA cup CSV file.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``all_cups_combined.csv``. Defaults to ``"data"``.
    mtime : float, optional
        Last modification time of the CSV file.  Included solely to invalidate
        the Streamlit cache when the file changes.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Date``, ``Competition``, ``HomeTeam`` and
        ``AwayTeam`` for matches where scores are missing.
    """

    data_dir = Path(data_dir)
    path = data_dir / CUP_FILE
    _ = mtime  # only used to bust Streamlit cache when the file changes
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
def load_cup_elo_tables(
    data_dir: str | Path = "data", mtime: float | None = None
) -> dict[str, pd.DataFrame]:
    """Return enriched ELO tables for each competition.

    The table for each competition includes:
        - current ELO rating
        - total points and goal stats
        - strength category based on ELO quantiles
        - average points per game against strong/average/weak opponents
        - average points per game at home and away
    """

    data_dir = Path(data_dir)
    path = data_dir / CUP_FILE
    _ = mtime  # only used to bust Streamlit cache when the file changes
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
        elo_values = list(elo_dict.values())
        p30, p70 = np.percentile(elo_values, [30, 70])

        def classify(e: float) -> str:
            if e >= p70:
                return "Strong"
            if e <= p30:
                return "Weak"
            return "Average"

        # initialize team stats containers
        stats: dict[str, dict] = {}
        for team, elo in elo_dict.items():
            stats[team] = {
                "elo": elo,
                "gf": 0,
                "ga": 0,
                "points": 0,
                "home_points": 0,
                "away_points": 0,
                "home_matches": 0,
                "away_matches": 0,
                "perf": {"Strong": [], "Average": [], "Weak": []},
            }

        for row in comp_df.itertuples(index=False):
            home, away = row.HomeTeam, row.AwayTeam
            hg, ag = row.FTHG, row.FTAG

            # home side
            h_pts = 3 if hg > ag else 1 if hg == ag else 0
            stats[home]["gf"] += hg
            stats[home]["ga"] += ag
            stats[home]["points"] += h_pts
            stats[home]["home_points"] += h_pts
            stats[home]["home_matches"] += 1
            opp_cat = classify(elo_dict.get(away, 1500))
            stats[home]["perf"][opp_cat].append(h_pts)

            # away side
            a_pts = 3 if ag > hg else 1 if ag == hg else 0
            stats[away]["gf"] += ag
            stats[away]["ga"] += hg
            stats[away]["points"] += a_pts
            stats[away]["away_points"] += a_pts
            stats[away]["away_matches"] += 1
            opp_cat = classify(elo_dict.get(home, 1500))
            stats[away]["perf"][opp_cat].append(a_pts)

        rows: list[dict] = []
        for team, s in stats.items():
            perf = s["perf"]
            vs_strong = round(sum(perf["Strong"]) / len(perf["Strong"]), 2) if perf["Strong"] else 0
            vs_avg = round(sum(perf["Average"]) / len(perf["Average"]), 2) if perf["Average"] else 0
            vs_weak = round(sum(perf["Weak"]) / len(perf["Weak"]), 2) if perf["Weak"] else 0
            home_ppg = (
                round(s["home_points"] / s["home_matches"], 2)
                if s["home_matches"]
                else 0
            )
            away_ppg = (
                round(s["away_points"] / s["away_matches"], 2)
                if s["away_matches"]
                else 0
            )
            rows.append(
                {
                    "Team": team,
                    "ELO": round(s["elo"], 1),
                    "Points": s["points"],
                    "GF": s["gf"],
                    "GA": s["ga"],
                    "GD": s["gf"] - s["ga"],
                    "Strength": classify(s["elo"]),
                    "PPG vs Strong": vs_strong,
                    "PPG vs Avg": vs_avg,
                    "PPG vs Weak": vs_weak,
                    "Home PPG": home_ppg,
                    "Away PPG": away_ppg,
                }
            )

        elo_table = pd.DataFrame(rows).sort_values("ELO", ascending=False).reset_index(drop=True)
        comp_name = COMPETITION_NAMES.get(comp_code, comp_code)
        tables[comp_name] = elo_table

    return tables


def render_uefa_cup_predictions(cross_league_df: pd.DataFrame, data_dir: str | Path = "data") -> None:
    """Render table of predicted outcomes for upcoming UEFA cup fixtures."""

    st.header("🏆 UEFA Cup Predictions")
    data_dir = Path(data_dir)
    cup_path = data_dir / CUP_FILE
    mtime = os.path.getmtime(cup_path) if cup_path.exists() else 0
    elo_tables = load_cup_elo_tables(data_dir, mtime)
    for competition, table in elo_tables.items():
        st.subheader(f"{competition} ELO Ratings")
        st.dataframe(table, hide_index=True, use_container_width=True)

    fixtures = load_upcoming_cup_fixtures(data_dir, mtime)
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
