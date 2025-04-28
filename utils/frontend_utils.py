import streamlit as st
import pandas as pd
from utils.poisson_utils import detect_risk_factors, detect_positive_factors


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
            return "❌ Krize"
        elif risk < 0.3 and positive > 0.6:
            return "✅ Nárůst formy"
        else:
            return "➖ Průměr"

    # Výpočty pro domácí tým
    risk_home, pos_home = detect_risk_factors(df, home_team, elo_dict)[1], detect_positive_factors(df, home_team, elo_dict)[1]
    overperformance_home, momentum_home = detect_overperformance_and_momentum(df, home_team)

    # Výpočty pro hostující tým
    risk_away, pos_away = detect_risk_factors(df, away_team, elo_dict)[1], detect_positive_factors(df, away_team, elo_dict)[1]
    overperformance_away, momentum_away = detect_overperformance_and_momentum(df, away_team)

    # Složení dat do tabulky
    df_status = pd.DataFrame({
        "Tým": [home_team, away_team],
        "Status": [status_label(risk_home, pos_home), status_label(risk_away, pos_away)],
        "Overperformance": [overperformance_home, overperformance_away],
        "Momentum": [momentum_home, momentum_away]
    })

    # Styling tabulky
    def style_status(val):
        if "Krize" in val:
            return 'color: red; font-weight: bold;'
        elif "Nárůst" in val:
            return 'color: green; font-weight: bold;'
        else:
            return 'color: gray;'

    styled = (
        df_status.style
        .applymap(style_status, subset=["Status"])
    )

    st.markdown("### 📊 Porovnání týmů")
    st.dataframe(styled, hide_index=True, use_container_width=True)


