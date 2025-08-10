import streamlit as st
import pandas as pd
from datetime import date
from io import BytesIO
from utils.poisson_utils import (
    load_data, calculate_elo_ratings,
    expected_goals_weighted_by_elo, poisson_prediction, match_outcomes_prob,
    over_under_prob, btts_prob, calculate_expected_points, get_top_scorelines,
    prob_to_odds
)
from utils.frontend_utils import validate_dataset


@st.cache_data
def get_league_data_and_elo(league_file: str):
    """Load league dataset and compute its ELO ratings with caching."""
    df_league = load_data(league_file)
    validate_dataset(df_league)
    elo_dict = calculate_elo_ratings(df_league)
    return df_league, elo_dict


def render_multi_match_predictions(session_state, home_team, away_team, league_name, league_file, league_files):
    st.title("📋 Hromadné predikce zápasů")

    if st.sidebar.button("➕ Přidat zápas"):
        if home_team != away_team:
            session_state.match_list.append({
                "league_file": league_file,
                "league_name": league_name,
                "home_team": home_team,
                "away_team": away_team
            })
        else:
            st.warning("⚠️ Vyber různé týmy!")

    if st.sidebar.button("🗑️ Vymazat všechny zápasy"):
        session_state.match_list.clear()

    if session_state.match_list:
        export_data = []

        for idx, match in enumerate(session_state.match_list):
            with st.container():
                st.markdown("---")
                st.subheader(f"🔮 {match['home_team']} vs {match['away_team']} {match['league_name']}")

                try:
                    df_match, elo_dict = get_league_data_and_elo(match["league_file"])
                    home_exp, away_exp = expected_goals_weighted_by_elo(
                        df_match, match["home_team"], match["away_team"], elo_dict
                    )
                    matrix = poisson_prediction(home_exp, away_exp)
                    outcomes = match_outcomes_prob(matrix)
                    over_under = over_under_prob(matrix, 2.5)
                    btts = btts_prob(matrix)
                    xpoints = calculate_expected_points(outcomes)

                    cols = st.columns(3)
                    cols[0].metric("⚽ Očekávané góly", f"{home_exp:.1f} - {away_exp:.1f}")
                    cols[1].metric("🔵 BTTS %", f"{btts['BTTS Yes']:.1f}%")
                    cols[2].metric("📈 Over 2.5 %", f"{over_under['Over 2.5']:.1f}%")
                    # Výpočet confidence score – rozdíl mezi nejvyšší a druhou nejvyšší výstupní pravděpodobností
                    sorted_probs = sorted(outcomes.values(), reverse=True)
                    confidence_index = round(sorted_probs[0] - sorted_probs[1], 1) if len(sorted_probs) >= 2 else 0.0

                    st.markdown("#### 🧠 Pravděpodobnosti výsledků")
                    result_cols = st.columns(4)
                    result_cols[0].metric("🏠 Výhra domácích", f"{outcomes['Home Win']:.1f}%", f"{prob_to_odds(outcomes['Home Win'])}")
                    result_cols[1].metric("🤝 Remíza", f"{outcomes['Draw']:.1f}%", f"{prob_to_odds(outcomes['Draw'])}")
                    result_cols[2].metric("🚶‍♂️ Výhra hostů", f"{outcomes['Away Win']:.1f}%", f"{prob_to_odds(outcomes['Away Win'])}")
                    result_cols[3].metric("🔒 Confidence", f"{confidence_index:.1f} %")
                    
                    top_scores = get_top_scorelines(matrix, top_n=1)
                    if top_scores:
                        top_score, top_prob = top_scores[0]
                        st.markdown(f"#### 🏅 Nejpravděpodobnější skóre: **{top_score[0]}:{top_score[1]}**")

                    if st.button(f"🗑️ Smazat zápas {match['home_team']} vs {match['away_team']}", key=f"del_{idx}"):
                        session_state.match_list.pop(idx)
                        st.rerun()

                    export_data.append({
                        "League": match["league_name"],
                        "Home": match["home_team"],
                        "Away": match["away_team"],
                        "Home ExpG": round(home_exp, 1),
                        "Away ExpG": round(away_exp, 1),
                        "BTTS %": round(btts['BTTS Yes'], 1),
                        "Over 2.5 %": round(over_under['Over 2.5'], 1),
                        "Home Win %": round(outcomes["Home Win"], 1),
                        "Draw %": round(outcomes["Draw"], 1),
                        "Away Win %": round(outcomes["Away Win"], 1),
                        "Top Score": f"{top_scores[0][0][0]}:{top_scores[0][0][1]}",
                        "Confidence %": confidence_index
                    })

                except Exception as e:
                    st.error(f"⚠️ Chyba při predikci: {str(e)}")

        if export_data:
            df_export = pd.DataFrame(export_data)
            today = date.today().strftime("%Y-%m-%d")
            num_matches = len(export_data)
            file_suffix = f"{today}_{num_matches:02d}"
            file_name = f"multi_predictions_{file_suffix}.xlsx"

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Predictions')

            st.download_button(
                label="📥 Stáhnout predikce jako Excel",
                data=output.getvalue(),
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("👈 Přidej zápasy přes tlačítko ➕ v sidebaru.")
