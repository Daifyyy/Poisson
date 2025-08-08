import streamlit as st
import pandas as pd
from utils.poisson_utils import detect_risk_factors, detect_positive_factors
from utils.poisson_utils.match_style import style_team_table


def validate_dataset(df):
    """Zkontroluje, zda dataset obsahuje potřebné sloupce."""
    required_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Datasetu chybí následující sloupce: {missing_columns}")


def display_team_status_table(home_team: str, away_team: str, df: pd.DataFrame, elo_dict: dict):
    from utils.poisson_utils import (
        detect_risk_factors,
        detect_positive_factors,
        detect_overperformance_and_momentum
    )

    def status_label(risk, positive):
        if risk > 0.6 and positive < 0.3:
            return "Krize"
        elif risk < 0.3 and positive > 0.6:
            return "Forma"
        else:
            return "Průměr"

    # Výpočty pro oba týmy
    risk_home, pos_home = detect_risk_factors(df, home_team, elo_dict)[1], detect_positive_factors(df, home_team, elo_dict)[1]
    overperformance_home, momentum_home = detect_overperformance_and_momentum(df, home_team)

    risk_away, pos_away = detect_risk_factors(df, away_team, elo_dict)[1], detect_positive_factors(df, away_team, elo_dict)[1]
    overperformance_away, momentum_away = detect_overperformance_and_momentum(df, away_team)

    # Tabulka
    df_status = pd.DataFrame({
        "Tým": [home_team, away_team],
        "Status": [status_label(risk_home, pos_home), status_label(risk_away, pos_away)],
        "Overperformance": [overperformance_home, overperformance_away],
        "Momentum": [momentum_home, momentum_away]
    })
    styled_df = style_team_table(df_status)
    st.markdown("### 📊 Porovnání týmů")
    st.dataframe(styled_df, hide_index=True, use_container_width=True)
    return styled_df


