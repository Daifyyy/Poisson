import streamlit as st
import pandas as pd
from utils.responsive import responsive_columns
from utils.poisson_utils import (
    calculate_form_emojis,
    calculate_expected_and_actual_points,
    aggregate_team_stats,
    calculate_team_pseudo_xg,
    add_btts_column,
    calculate_conceded_goals,
    calculate_recent_team_form,
    calculate_elo_changes,
    calculate_team_styles,
    intensity_score_to_emoji,
    compute_score_stats,
    compute_form_trend,
    calculate_strength_of_schedule,
)
from utils.statistics import calculate_clean_sheets


def render_league_overview(season_df, league_name, gii_dict, elo_dict):
    query_params = st.query_params
    if "selected_team" in query_params:
        return  # Pokud je aktivní detail týmu, sekce overview se nespustí

    st.header(f"🏆 {league_name}")

    num_matches = len(season_df)
    if num_matches == 0:
        st.info("No match data available for this league.")
        return

    avg_goals = round((season_df["FTHG"] + season_df["FTAG"]).mean(), 1)
    season_df = add_btts_column(season_df)
    btts_pct = round(100 * season_df["BTTS"].mean(), 1)
    over_25 = round(
        100
        * season_df[(season_df["FTHG"] + season_df["FTAG"]) > 2.5].shape[0]
        / num_matches,
        1,
    )

    st.markdown(
        f"🗕️ Zápasů: {num_matches}	⚽ Průměr gólů: {avg_goals}	🤽 BTTS: {btts_pct}%	📈 Over 2.5: {over_25}%"
    )

    form_emojis = calculate_form_emojis(season_df)
    points_data = calculate_expected_and_actual_points(season_df)
    team_stats = aggregate_team_stats(season_df)
    over25 = (
        season_df.groupby("HomeTeam")
        .apply(lambda x: (x["FTHG"] + x["FTAG"] > 2.5).mean() * 100)
        .round(0)
    )
    btts = season_df.groupby("HomeTeam")["BTTS"].mean().mul(100).round(0)
    xg_stats = calculate_team_pseudo_xg(season_df)
    sos_dict = calculate_strength_of_schedule(season_df, metric="elo")

    trends = []
    avg_goals_all = []
    score_var = []

    for team in team_stats.index:
        score_list, avg_goals_per_match, score_variance = compute_score_stats(
            season_df, team
        )
        trends.append(compute_form_trend(score_list))
        avg_goals_all.append(round(avg_goals_per_match, 1))
        score_var.append(round(score_variance, 1))

    summary_table = pd.DataFrame(
        {
            "Tým": team_stats.index,
            "Elo": pd.Series(team_stats.index.map(lambda t: elo_dict.get(t, 1500)))
            .round(0)
            .values,
            "SOS": pd.Series(team_stats.index.map(lambda t: sos_dict.get(t, 0)))
            .round(1)
            .values,
            "Body": team_stats.index.map(
                lambda t: points_data.get(t, {}).get("points", 0)
            ),
            "Form": team_stats.index.map(lambda t: form_emojis.get(t, "❄️❄️❄️")),
            "Trend formy": trends,
            "Góly celkem": avg_goals_all,
            "Rozptyl skóre": score_var,
            "Vstřelené Góly": team_stats["Góly"].round(1),
            "Střely": team_stats["Střely"].round(1),
            "Na branku": team_stats["Na branku"].round(1),
            "Rohy": team_stats["Rohy"].round(1),
            "Obdržené góly": team_stats["Obdržené góly"].round(1),
            "Čistá konta %": team_stats.index.map(
                lambda t: f"{calculate_clean_sheets(season_df, t)}%"
            ),
            "Over 2.5 %": team_stats.index.map(over25).astype(str) + "%",
            "BTTS %": team_stats.index.map(btts).astype(str) + "%",
            "Intenzita": team_stats.index.map(
                lambda t: intensity_score_to_emoji(gii_dict.get(t))
            ),
        }
    )

    sort_option = st.selectbox("Seřadit podle:", ["Body", "SOS"], index=0)
    if sort_option == "SOS":
        summary_table = summary_table.sort_values("SOS", ascending=False)
    else:
        summary_table = summary_table.sort_values("Body", ascending=False)
    summary_table = summary_table.reset_index(drop=True)
    import urllib.parse
    import urllib

    def clickable_team_link(team):
        encoded_team = urllib.parse.quote_plus(team)
        encoded_league = urllib.parse.quote_plus(league_name)
        return f'<a href="?selected_team={encoded_team}&selected_league={encoded_league}">🔍 {team}</a>'

    summary_table_display = summary_table.copy()

    top_idx = summary_table["SOS"].nlargest(3).index
    bottom_idx = summary_table["SOS"].nsmallest(3).index
    summary_table_display["SOS"] = summary_table_display["SOS"].round(1).astype(str)
    summary_table_display.loc[top_idx, "SOS"] += " 🔥"
    summary_table_display.loc[bottom_idx, "SOS"] += " 🍀"

    st.markdown(
        """
    <style>
        .stDataFrame tbody tr td:first-child { white-space: nowrap; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("Kliknutím na tým zobrazíš jeho detail:")
    float_cols = summary_table_display.select_dtypes(include="float").columns
    format_dict = {col: "{:.1f}" for col in float_cols}
    format_dict["Tým"] = clickable_team_link
    styled_table = summary_table_display.style.format(format_dict, escape=None)
    st.markdown(styled_table.to_html(escape=False), unsafe_allow_html=True)

    st.markdown("### 🌟 Top 5 týmy")
    cols = responsive_columns(4)
    cols[0].markdown("🔮 **Nejvíc gólů**")
    cols[0].dataframe(
        summary_table.sort_values("Vstřelené Góly", ascending=False).head(5)[
            ["Tým", "Vstřelené Góly"]
        ],
        hide_index=True,
    )

    conceded_df = calculate_conceded_goals(season_df)
    cols[1].markdown("🔴 **Nejvíce obdržených gólů**")
    cols[1].dataframe(conceded_df.head(5), hide_index=True)

    form_df = calculate_recent_team_form(season_df)
    cols[2].markdown("📉 **Nejhorší forma**")
    cols[2].dataframe(form_df.head(5)[["Tým", "Form"]], hide_index=True)

    cols[3].markdown("📈 **Nejlepší forma**")
    cols[3].dataframe(
        summary_table.sort_values("Form", ascending=False).head(5)[["Tým", "Form"]],
        hide_index=True,
    )

    st.markdown("### 🔧 Styl a vývoj týmů")
    elo_df = calculate_elo_changes(season_df)
    elo_drop_df = elo_df.sort_values("Změna").head(5).reset_index(drop=True)
    off_df, def_df = calculate_team_styles(season_df)

    cols2 = responsive_columns(4)
    cols2[0].markdown("📈 **ELO zlepšení**")
    cols2[0].dataframe(elo_df.head(5), hide_index=True)
    cols2[1].markdown("📉 **ELO poklesy**")
    cols2[1].dataframe(elo_drop_df, hide_index=True)
    cols2[2].markdown("⚡ **Ofenzivní styl**")
    cols2[2].dataframe(off_df.head(5)[["Tým", "Ofenzivní styl index"]], hide_index=True)
    cols2[3].markdown("🧱 **Defenzivní styl**")
    cols2[3].dataframe(
        def_df.head(5)[["Tým", "Defenzivní styl index"]], hide_index=True
    )
