# Nová funkce pro výstup dat do Excelu
# Umístit např. do `match_prediction_section.py` nebo oddělit jako `export_utils.py`

import pandas as pd
import streamlit as st
from io import BytesIO

def generate_excel_analysis_export(
    league_name, home_team, away_team,
    expected_score, outcomes, over_under, btts,
    xpoints, xg_home, xg_away,
    expected_tempo, tempo_rating,
    warnings_home, warnings_away,
    positives_home, positives_away,
    team_stats, style_home, style_away,
    form_home, form_away,
    h2h_stats, top_scorelines, goal_probs,
    variance_warning, style_form_warning, style_conflict_warning
):
    """
    Výstup všech analytických informací do Excel souboru.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

        # 1. Základní info
        df_main = pd.DataFrame({
            "League": [league_name],
            "Home": [home_team],
            "Away": [away_team],
            "Expected Home Goals": [expected_score[0]],
            "Expected Away Goals": [expected_score[1]],
            "Home Win %": [outcomes['Home Win']],
            "Draw %": [outcomes['Draw']],
            "Away Win %": [outcomes['Away Win']],
            "BTTS %": [btts['BTTS Yes']],
            "Over 2.5 %": [over_under['Over 2.5']],
            "Home xP": [xpoints['Home xP']],
            "Away xP": [xpoints['Away xP']],
            "xG Home": [xg_home['xG_home']],
            "xG Away": [xg_away['xG_away']],
            "Tempo (rating)": [tempo_rating],
            "Tempo (value)": [expected_tempo],
        })
        df_main.to_excel(writer, index=False, sheet_name="Match Overview")

        # 2. Warningy a trendy
        df_warn = pd.DataFrame({
            "Team": [home_team, away_team],
            "Warnings": ["; ".join(warnings_home), "; ".join(warnings_away)],
            "Positive Trends": ["; ".join(positives_home), "; ".join(positives_away)]
        })
        df_warn.to_excel(writer, index=False, sheet_name="Warnings & Trends")

        # 3. Statistické porovnání
        pd.DataFrame(team_stats).T.to_excel(writer, sheet_name="Team Stats")

        # 4. Styl hry
        df_style = pd.DataFrame({
            "Attribute": ["Tempo", "Aggressiveness", "Dominance"],
            home_team: [style_home['rating'], style_home['aggressiveness_rating'], style_home['imbalance_type']],
            away_team: [style_away['rating'], style_away['aggressiveness_rating'], style_away['imbalance_type']],
        })
        df_style.to_excel(writer, index=False, sheet_name="Match Style")

        # 5. Formy proti silným/průměrným/slabým
        form_home.T.to_excel(writer, sheet_name=f"{home_team} Form")
        form_away.T.to_excel(writer, sheet_name=f"{away_team} Form")

        # 6. Head-to-Head
        pd.DataFrame([h2h_stats]).to_excel(writer, sheet_name="Head2Head", index=False)

        # 7. Top scorelines
        pd.DataFrame([
            {"Scoreline": f"{a}:{b}", "Probability": f"{round(p * 100, 1)} %"}
            for (a, b), p in top_scorelines
        ]).to_excel(writer, sheet_name="Top Scorelines", index=False)

        # 8. Pravděpodobnosti gólů
        goal_probs.to_excel(writer, sheet_name="Goal Probabilities", index=False)

        # 9. Dodatečná varování
        extra = pd.DataFrame({
            "Type": ["Scoreline Variance", "Style+Form", "Conflict Style"],
            "Message": [variance_warning or "", style_form_warning or "", style_conflict_warning or ""]
        })
        extra.to_excel(writer, index=False, sheet_name="Extra Warnings")

    output.seek(0)
    return output


# Ve `render_single_match_prediction` přidej na konec sekce tlačítko:
# if st.button("📥 Stáhnout analytickou zprávu jako Excel"):
#     excel_file = generate_excel_analysis_export(...)  # doplň argumenty
#     st.download_button("📥 Stáhnout report", data=excel_file, file_name=f"{home_team}_vs_{away_team}_analysis.xlsx")