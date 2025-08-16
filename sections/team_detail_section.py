import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

import plotly.graph_objects as go

from utils.responsive import responsive_columns
from utils.poisson_utils import (
    elo_history, calculate_form_emojis, calculate_expected_and_actual_points,
    aggregate_team_stats, calculate_team_pseudo_xg, get_whoscored_xg_xga,
    calculate_conceded_goals, calculate_recent_team_form,
    calculate_elo_changes, calculate_team_styles,
    intensity_score_to_emoji, compute_score_stats, compute_form_trend,
    merged_home_away_opponent_form, classify_team_strength, calculate_advanced_team_metrics,
    calculate_team_extra_stats, get_team_record, analyze_team_profile, generate_team_comparison,
    render_team_comparison_section
)
from utils.poisson_utils.team_analysis import TEAM_COMPARISON_ICON_MAP, TEAM_COMPARISON_DESC_MAP
from utils.form_trend import get_rolling_form


def render_team_detail(
    _df: pd.DataFrame,
    season_df: pd.DataFrame,
    team: str,
    league_name: str,
    gii_dict: Dict[str, float],
) -> None:
    """Render detailed information for a selected team.

    Args:
        _df: Unused historical dataset (kept for backward compatibility).
        season_df: DataFrame with season data for statistics.
        team: Team name to display.
        league_name: Name of the league.
        gii_dict: Mapping of teams to intensity index values.

    Returns:
        None
    """
    st.sidebar.markdown("### ⏱️ Časový filtr")

    time_filter = st.sidebar.radio(
        "Vyber rozsah dat",
        ["Celá sezóna", "Posledních 5 zápasů", "Posledních 10 zápasů", "Posledních 5 doma", "Posledních 5 venku"]
    )

    def _apply_time_filter(data: pd.DataFrame, team_name: str) -> pd.DataFrame:
        df_sorted = data.sort_values("Date")
        df_sorted["DateDiff"] = df_sorted["Date"].diff().dt.days
        gap_threshold = 30
        latest_index = df_sorted[df_sorted["Date"] == df_sorted["Date"].max()].index[0]
        cutoff_idx = df_sorted.iloc[:latest_index].loc[df_sorted["DateDiff"] > gap_threshold].last_valid_index()
        season_start = df_sorted.loc[cutoff_idx + 1, "Date"] if cutoff_idx is not None else df_sorted["Date"].min()
        season_cutoff = data[data['Date'] >= season_start]

        if time_filter == "Posledních 5 zápasů":
            matches = season_cutoff[(season_cutoff['HomeTeam'] == team_name) | (season_cutoff['AwayTeam'] == team_name)]
            return matches.sort_values("Date", ascending=False).head(5)
        if time_filter == "Posledních 10 zápasů":
            matches = season_cutoff[(season_cutoff['HomeTeam'] == team_name) | (season_cutoff['AwayTeam'] == team_name)]
            return matches.sort_values("Date", ascending=False).head(10)
        if time_filter == "Posledních 5 doma":
            matches = season_cutoff[season_cutoff['HomeTeam'] == team_name]
            return matches.sort_values("Date", ascending=False).head(5)
        if time_filter == "Posledních 5 venku":
            matches = season_cutoff[season_cutoff['AwayTeam'] == team_name]
            return matches.sort_values("Date", ascending=False).head(5)
        return season_cutoff

    original_df = season_df
    filtered_df = _apply_time_filter(original_df, team)

    difficulty_filter = st.sidebar.selectbox(
        "🎯 Filtrovat podle síly soupeře:",
        ["Vše", "Silní", "Průměrní", "Slabí"]
    )

    teams_home = original_df["HomeTeam"].unique().tolist()
    teams_away = original_df["AwayTeam"].unique().tolist()
    compare_options = sorted(set(teams_home + teams_away) - {team})
    compare_team = st.sidebar.selectbox(
        "🔄 Porovnat s jiným týmem:",
        ["Žádný"] + compare_options
    )

    def _apply_difficulty_filter(data: pd.DataFrame) -> pd.DataFrame:
        if difficulty_filter != "Vše":
            data = data.copy()
            data["Opponent"] = data.apply(lambda row: row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam'], axis=1)
            data["Soupeř síla"] = data["Opponent"].apply(lambda opp: classify_team_strength(filtered_df, opp))
            data = data[data["Soupeř síla"] == difficulty_filter]
        return data

    season_df = _apply_difficulty_filter(filtered_df)

    home = season_df[season_df['HomeTeam'] == team]
    away = season_df[season_df['AwayTeam'] == team]
    all_matches = pd.concat([home, away])

    if compare_team and compare_team != "Žádný" and compare_team != team:
        compare_df = _apply_time_filter(original_df, compare_team)
        compare_home = compare_df[compare_df['HomeTeam'] == compare_team]
        compare_away = compare_df[compare_df['AwayTeam'] == compare_team]
        compare_matches = pd.concat([compare_home, compare_away])

        total_df = pd.concat([all_matches, compare_matches])
        home_df = pd.concat([home, compare_away])
        away_df = pd.concat([away, compare_home])

        stats_total = generate_team_comparison(total_df, team, compare_team)
        stats_home = generate_team_comparison(home_df, team, compare_team)
        stats_away = generate_team_comparison(away_df, team, compare_team)

        if stats_total.empty:
            st.warning("⚠️ Jeden z týmů nemá dostupná data pro zvolený filtr.")
            return

        render_team_comparison_section(
            team, compare_team,
            stats_total, stats_home, stats_away
        )
        st.divider()
        footer_cols = st.columns(4)

        fig = go.Figure()

        team_trend = get_rolling_form(all_matches, team)
        fig.add_trace(
            go.Scatter(
                x=team_trend["Date"],
                y=team_trend["rolling_points"],
                mode="lines",
                name=f"{team} body",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=team_trend["Date"],
                y=team_trend["rolling_xg"],
                mode="lines",
                name=f"{team} xG",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=team_trend["Date"],
                y=team_trend["ELO"],
                mode="lines",
                name=f"{team} ELO",
                yaxis="y2",
            )
        )

        compare_trend = get_rolling_form(compare_matches, compare_team)
        fig.add_trace(
            go.Scatter(
                x=compare_trend["Date"],
                y=compare_trend["rolling_points"],
                mode="lines",
                name=f"{compare_team} body",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=compare_trend["Date"],
                y=compare_trend["rolling_xg"],
                mode="lines",
                name=f"{compare_team} xG",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=compare_trend["Date"],
                y=compare_trend["ELO"],
                mode="lines",
                name=f"{compare_team} ELO",
                yaxis="y2",
                line=dict(dash="dash"),
            )
        )

        fig.update_layout(
            yaxis=dict(title="Rolling body/xG"),
            yaxis2=dict(title="ELO", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )

        footer_cols[0].plotly_chart(fig, use_container_width=True)

        return






    st.header(f"📌 Detail týmu: {team}")

    # Výpočet pro všechny tři varianty
    # Výpočty
    record_all = get_team_record(season_df, team)
    record_home = get_team_record(season_df, team, side="home")
    record_away = get_team_record(season_df, team, side="away")

    # Výpis bilance
    st.markdown(
        f"**📊 Bilance:** &nbsp;&nbsp;&nbsp;"
        f"🟦 Celkem – ✅ {record_all[0]} | 🤝 {record_all[1]} | ❌ {record_all[2]} &nbsp;&nbsp;&nbsp;"
        f"🏠 Doma – ✅ {record_home[0]} | 🤝 {record_home[1]} | ❌ {record_home[2]} &nbsp;&nbsp;&nbsp;"
        f"🚌 Venku – ✅ {record_away[0]} | 🤝 {record_away[1]} | ❌ {record_away[2]}",
        unsafe_allow_html=True
    )

    # Sezónní xG a xGA – primárně z WhoScored, fallback na pseudo-xG
    ws_stats = get_whoscored_xg_xga(team)
    pseudo_stats = calculate_team_pseudo_xg(season_df).get(team, {})

    team_xg = ws_stats.get("xg", np.nan)
    team_xga = ws_stats.get("xga", np.nan)
    if np.isnan(team_xg):
        team_xg = pseudo_stats.get("xg", 0)
    if np.isnan(team_xga):
        team_xga = pseudo_stats.get("xga", 0)

    col_xg, col_xga = st.columns(2)
    col_xg.metric("Sezónní xG", f"{team_xg:.1f}")
    col_xga.metric("Sezónní xGA", f"{team_xga:.1f}")

    team_stats = aggregate_team_stats(season_df)
    if team not in team_stats.index:
        st.error(f"Tým '{team}' nebyl nalezen v datech. Zkontroluj správnost názvu.")
        st.stop()

    
    # Ligový průměr
    league_avg = team_stats.mean()
    def compare_stat(name, team_value, league_avg):
        league_value = league_avg.get(name, 0)
        diff = team_value - league_value
        return f" *(liga: {league_value:.1f}, Δ {diff:+.1f})*"

    advanced_stats = calculate_advanced_team_metrics(season_df)
    league_avg_advanced = advanced_stats.mean()

    
    
    
    stats = team_stats.loc[team]
    #card_stats = get_team_card_stats(season_df, team)
    # stats['Žluté'] = card_stats['yellow']
    # stats['Červené'] = card_stats['red']

    # ✅ Kontrola rozsahu dat a počtu zápasů
    st.caption(f"Počet zápasů v aktuálním datasetu: {len(season_df)}")
    st.caption(f"Rozsah dat: {season_df['Date'].min().date()} až {season_df['Date'].max().date()}")

    # Fallback values for home/away splits using pseudo-xG
    def _pseudo_xg_split(df_home: pd.DataFrame, df_away: pd.DataFrame):
        coeff_shot = 0.1
        coeff_on = 0.3
        xg_h = (
            (df_home['HS'] * coeff_shot + df_home['HST'] * coeff_on).mean()
            if not df_home.empty else 0.0
        )
        xga_h = (
            (df_home['AS'] * coeff_shot + df_home['AST'] * coeff_on).mean()
            if not df_home.empty else 0.0
        )
        xg_a = (
            (df_away['AS'] * coeff_shot + df_away['AST'] * coeff_on).mean()
            if not df_away.empty else 0.0
        )
        xga_a = (
            (df_away['HS'] * coeff_shot + df_away['HST'] * coeff_on).mean()
            if not df_away.empty else 0.0
        )
        return xg_h, xga_h, xg_a, xga_a

    home_xg, home_xga, away_xg, away_xga = _pseudo_xg_split(home, away)

    #card_stats = get_team_card_stats(season_df, team)
    # yellow_per_foul = card_stats["yellow"] / card_stats["fouls"] if card_stats["fouls"] else 0
    # red_per_foul = card_stats["red"] / card_stats["fouls"] if card_stats["fouls"] else 0

    # ✅ Očekávané body
    xp_data = calculate_expected_and_actual_points(season_df).get(team, {})
    expected_points = xp_data.get("expected_points", 0)

    

    def calc_metrics(df, is_home=None):
        if is_home is None:  # All matches
            goals = pd.concat([df[df['HomeTeam'] == team]['FTHG'], df[df['AwayTeam'] == team]['FTAG']]).mean()
            conceded = pd.concat([df[df['HomeTeam'] == team]['FTAG'], df[df['AwayTeam'] == team]['FTHG']]).mean()
            shots = pd.concat([df[df['HomeTeam'] == team]['HS'], df[df['AwayTeam'] == team]['AS']]).mean()
            shots_on = pd.concat([df[df['HomeTeam'] == team]['HST'], df[df['AwayTeam'] == team]['AST']]).mean()
            corners = pd.concat([df[df['HomeTeam'] == team]['HC'], df[df['AwayTeam'] == team]['AC']]).mean()
            fouls = pd.concat([df[df['HomeTeam'] == team]['HF'], df[df['AwayTeam'] == team]['AF']]).mean()
            yellow = pd.concat([df[df['HomeTeam'] == team]['HY'], df[df['AwayTeam'] == team]['AY']]).mean()
            red = pd.concat([df[df['HomeTeam'] == team]['HR'], df[df['AwayTeam'] == team]['AR']]).mean()
        else:
            goals = df['FTHG'].mean() if is_home else df['FTAG'].mean()
            conceded = df['FTAG'].mean() if is_home else df['FTHG'].mean()
            shots = df['HS'].mean() if is_home else df['AS'].mean()
            shots_on = df['HST'].mean() if is_home else df['AST'].mean()
            corners = df['HC'].mean() if is_home else df['AC'].mean()
            fouls = df['HF'].mean() if is_home else df['AF'].mean()
            yellow = df['HY'].mean() if is_home else df['AY'].mean()
            red = df['HR'].mean() if is_home else df['AR'].mean()

        return {
            "Góly": goals,
            "Obdržené góly": conceded,
            "Střely": shots,
            "Na branku": shots_on,
            "Rohy": corners,
            "Fauly": fouls,
            "Žluté": yellow,
            "Červené": red,
        }

    # Výpočet metrik a ligového průměru
    metrics_all = calc_metrics(all_matches)
    metrics_home = calc_metrics(home, is_home=True)
    metrics_away = calc_metrics(away, is_home=False)

    extra_all = calculate_team_extra_stats(all_matches, team)
    extra_home = calculate_team_extra_stats(home, team)
    extra_away = calculate_team_extra_stats(away, team)


    st.markdown("### 📊 Průměrné statistiky – Celkem / Doma / Venku")

    home_adv = calculate_advanced_team_metrics(home)
    away_adv = calculate_advanced_team_metrics(away)

    def adv_value(df, metric):
        return df.loc[team, metric] * 100 if not df.empty and team in df.index else 0.0

    data_table = {
        "Celkem": {
            "xG": team_xg,
            "xGA": team_xga,
            **metrics_all,
            "Přesnost střel %": adv_value(advanced_stats, "Přesnost střel"),
            "Konverzní míra %": adv_value(advanced_stats, "Konverzní míra"),
            "Čistá konta %": extra_all["Čistá konta %"],
            "BTTS %": extra_all["BTTS %"],
        },
        "Doma": {
            "xG": home_xg,
            "xGA": home_xga,
            **metrics_home,
            "Přesnost střel %": adv_value(home_adv, "Přesnost střel"),
            "Konverzní míra %": adv_value(home_adv, "Konverzní míra"),
            "Čistá konta %": extra_home["Čistá konta %"],
            "BTTS %": extra_home["BTTS %"],
        },
        "Venku": {
            "xG": away_xg,
            "xGA": away_xga,
            **metrics_away,
            "Přesnost střel %": adv_value(away_adv, "Přesnost střel"),
            "Konverzní míra %": adv_value(away_adv, "Konverzní míra"),
            "Čistá konta %": extra_away["Čistá konta %"],
            "BTTS %": extra_away["BTTS %"],
        },
    }

    metrics_df = pd.DataFrame(data_table)
    metrics_df = metrics_df.reindex([
        "xG",
        "xGA",
        "Góly",
        "Obdržené góly",
        "Střely",
        "Na branku",
        "Rohy",
        "Fauly",
        "Žluté",
        "Červené",
        "Přesnost střel %",
        "Konverzní míra %",
        "Čistá konta %",
        "BTTS %",
    ])

    icon_map = TEAM_COMPARISON_ICON_MAP.copy()
    icon_map["xGA"] = "🚫"
    icon_map["Přesnost střel %"] = icon_map.pop("Přesnost střel", "")
    icon_map["Konverzní míra %"] = icon_map.pop("Konverzní míra", "")

    display_df = metrics_df.round(1)
    display_df.index = [f"{icon_map.get(idx, '')} {idx}" for idx in display_df.index]

    with st.expander("Legenda"):
        desc_map = TEAM_COMPARISON_DESC_MAP.copy()
        desc_map["xGA"] = "Očekávané obdržené góly podle WhoScored"
        desc_map["Přesnost střel %"] = desc_map.pop("Přesnost střel", "")
        desc_map["Konverzní míra %"] = desc_map.pop("Konverzní míra", "")
        for key in metrics_df.index:
            icon = icon_map.get(key, "")
            desc = desc_map.get(key, "")
            st.markdown(f"{icon} {key} – {desc}")

    st.table(display_df.style.format("{:.1f}"))

    st.markdown("---")

    
    
    # ✅ Připrav zápasy týmu
    df_team = season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)].copy()

    # Přidat info o soupeři
    df_team['Opponent'] = df_team.apply(lambda row: row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam'], axis=1)
    df_team['H/A'] = df_team.apply(lambda row: 'H' if row['HomeTeam'] == team else 'A', axis=1)

    # ✅ Kategorizace síly soupeře
    df_team['Soupeř síla'] = df_team['Opponent'].apply(lambda opp: classify_team_strength(season_df, opp))

    # ✅ Aplikace filtru podle obtížnosti
    if difficulty_filter != "Vše":
        df_team = df_team[df_team["Soupeř síla"] == difficulty_filter]

    # Remove matches without a final score to avoid processing upcoming fixtures
    df_team = df_team.dropna(subset=["FTHG", "FTAG"])

    # Posledních 5 zápasů
    last_matches = df_team.sort_values("Date", ascending=False).head(5)

    # ✅ Formátování do tabulky
    def format_result(row):
        is_home = row['HomeTeam'] == team
        opponent = row['AwayTeam'] if is_home else row['HomeTeam']
        team_goals = row['FTHG'] if is_home else row['FTAG']
        opp_goals = row['FTAG'] if is_home else row['FTHG']
        return pd.Series({
            "Datum": row['Date'].date(),
            "Soupeř": opponent,  # ❌ žádný prefix
            "H/A": "H" if is_home else "A",
            "Skóre": f"{team_goals}:{opp_goals}",
            "Střely": row['HS'] if is_home else row['AS'],
            "Na branku": row['HST'] if is_home else row['AST'],
            "Fauly": row['HF'] if is_home else row['AF'],
            "Žluté": row['HY'] if is_home else row['AY'],
            "Červené": row['HR'] if is_home else row['AR'],
        })

    # ✅ Převod a styling
    match_details = last_matches.apply(format_result, axis=1)
    match_details = match_details.reset_index(drop=True)  # ✅ odstraní indexový sloupec

    def highlight_result(row):
        score = row["Skóre"].split(":")
        if len(score) != 2 or not all(part.isdigit() for part in score):
            return [""] * len(row)
        team_goals, opp_goals = map(int, score)
        color = "#d4edda" if team_goals > opp_goals else "#f8d7da" if team_goals < opp_goals else "#fff3cd"
        return [f"background-color: {color}"] * len(row)

    styled_matches = match_details.style.apply(highlight_result, axis=1).format(precision=1)

    # ✅ Výstup
    st.markdown("### 🕵️ Posledních 5 zápasů")
    # st.dataframe(
    #     styled_matches.hide(axis="index").set_table_attributes('style="width: 100%;"').set_table_styles([
    #         {"selector": "th", "props": [("text-align", "left")]}
    #     ]),
    #     use_container_width=True
    # )
    st.table(styled_matches)

    # Disciplinovanost – karty na faul
    yellow_per_foul = stats['Žluté'] / stats['Fauly'] if stats['Fauly'] else 0
    red_per_foul = stats.get('Červené', 0) / stats['Fauly'] if stats['Fauly'] else 0

    # # Defenzivní efektivita – gól na střelu
    defensive_efficiency = (stats['Obdržené góly'] / stats['Střely']) if stats['Střely'] else 0

    # # Přesnost a konverze
    conversion_rate = (stats['Góly'] / stats['Střely']) if stats['Střely'] else 0

    df_team = season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)].copy()
    df_team['Opponent'] = df_team.apply(lambda row: row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam'], axis=1)
    df_team['Strength'] = df_team['Opponent'].apply(lambda t: classify_team_strength(season_df, t))

    profile = analyze_team_profile(
        season_df,
        team,
        conversion_rate,
        defensive_efficiency,
        yellow_per_foul,
        red_per_foul
    )
    
    st.markdown("---")
    
    if (
        profile["výherní série"] >= 2 or
        profile["proherní série"] >= 2 or
        profile["silné stránky"] != "Není výrazná" or
        profile["rizika"] != "Bez zásadních slabin" or
        profile["profilové hodnocení"]
    ):
        st.markdown("### 📊 Zhodnocení týmu")
        st.markdown(f"- 🔥 Aktuální forma: **{profile['forma']}**")
        if profile["výherní série"] >= 2:
            st.markdown(f"- 🏆 Výherní série: **{profile['výherní série']}** zápasy")
        if profile["proherní série"] >= 2:
            st.markdown(f"- ❌ Série proher: **{profile['proherní série']}**")
        if profile["bez čistého konta"] >= 3:
            st.markdown(f"- 🚫 Bez čistého konta: **{profile['bez čistého konta']}** zápasů")
        if profile["silné stránky"] != "Není výrazná":
            st.markdown(f"- 💪 Silné stránky: {profile['silné stránky']}")
        if profile["rizika"] != "Bez zásadních slabin":
            st.markdown(f"- ⚠️ Rizika: {profile['rizika']}")
        st.markdown(f"- 🎯 Styl týmu: {profile['styl']}")
        if profile["profilové hodnocení"]:
            st.markdown("### 🧩 Další pozorování")
            for tag in profile["profilové hodnocení"]:
                st.markdown(f"- {tag}")


    # 📈 ELO rating progression
    elo_prog = elo_history(season_df, team)
    if not elo_prog.empty:
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        ax.plot(elo_prog["Date"], elo_prog["ELO"], marker="o")
        ax.set_title("Vývoj ELO ratingu")
        ax.set_xlabel("Datum")
        ax.set_ylabel("ELO")
        plt.xticks(rotation=45)
        cols = responsive_columns(4)
        cols[0].pyplot(fig)

    st.divider()
    footer_cols = st.columns(4)

    form_trend = get_rolling_form(all_matches, team)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=form_trend["Date"],
            y=form_trend["rolling_points"],
            mode="lines",
            name="Body",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=form_trend["Date"],
            y=form_trend["rolling_xg"],
            mode="lines",
            name="xG",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=form_trend["Date"],
            y=form_trend["ELO"],
            mode="lines",
            name="ELO",
            yaxis="y2",
        )
    )

    fig.update_layout(
        yaxis=dict(title="Rolling body/xG"),
        yaxis2=dict(title="ELO", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )

    footer_cols[0].plotly_chart(fig, use_container_width=True)

    return

    
    # def extract_match_stats(row):
    #     is_home = row['HomeTeam'] == team
    #     return pd.Series({
    #         "Soupeř": row['Opponent'],
    #         "Typ soupeře": row['Strength'],
    #         "Góly": row['FTHG'] if is_home else row['FTAG'],
    #         "Střely": row['HS'] if is_home else row['AS'],
    #         "Na branku": row['HST'] if is_home else row['AST'],
    #         "Fauly": row['HF'] if is_home else row['AF'],
    #         "Žluté": row['HY'] if is_home else row['AY'],
    #         "Červené": row['HR'] if is_home else row['AR']
    #     })

    # detailed_stats = df_team.apply(extract_match_stats, axis=1)
    # numeric_columns = detailed_stats.select_dtypes(include='number').columns
    # avg_by_strength = detailed_stats.groupby("Typ soupeře")[numeric_columns].mean().round(2)
    # st.markdown("---")
    # st.subheader("📉 Výkonnost proti kategoriím soupeřů")
    # st.markdown("Souhrnné statistiky proti různě silným soupeřům")
    # # Shrnutí disciplíny
    # if yellow_per_foul > 0.25:
    #     st.markdown(f"🟡 Tým fauluje poměrně neukázněně – **{yellow_per_foul:.2f}** žlutých na 1 faul.")
    # else:
    #     st.markdown(f"🟢 Tým je relativně disciplinovaný – **{yellow_per_foul:.2f}** žlutých na 1 faul.")

    # if red_per_foul > 0.05:
    #     st.markdown(f"🔴 Relativně vysoký výskyt červených karet: **{red_per_foul:.2f}** na faul.")

    # # Shrnutí konverze a obrany
    # if conversion_rate > 0.15 and defensive_efficiency > 0.12:
    #     st.markdown("🔁 **Souhrn:** Tým má ofenzivní sílu, ale defenzivní slabiny.")
    # elif conversion_rate < 0.08 and defensive_efficiency < 0.07:
    #     st.markdown("🧤 **Souhrn:** Tým je defenzivně pevný, ale v útoku neefektivní.")
    # elif conversion_rate > 0.15 and defensive_efficiency < 0.07:
    #     st.markdown("💪 **Souhrn:** Tým je dominantní na obou stranách – silný útok i obrana.")
    # elif conversion_rate < 0.08 and defensive_efficiency > 0.12:
    #     st.markdown("⚠️ **Souhrn:** Tým má potíže v útoku i v obraně.")

    # if conversion_rate > 0.15:
    #     st.markdown(f"⚽ Tým má vysokou konverzní míru – **{conversion_rate*100:.1f}%** střel končí gólem.")
    # elif conversion_rate < 0.08:
    #     st.markdown(f"🚫 Nízká konverzní míra – pouze **{conversion_rate*100:.1f}%** střel je gólových.")

    # if defensive_efficiency > 0.12:
    #     st.markdown(f"❗ Tým dostává gól z každé 8. střely – defenziva je zranitelná.")
    # st.table(avg_by_strength.style.format("{:.2f}"))

    # # Verbální shrnutí výkonu proti kategoriím soupeřů
    # if set(["Silný", "Slabý"]).issubset(avg_by_strength.index):
    #     g_strong = avg_by_strength.loc["Silný", "Góly"]
    #     g_weak = avg_by_strength.loc["Slabý", "Góly"]
    #     d_strong = avg_by_strength.loc["Silný", "Na branku"]
    #     d_weak = avg_by_strength.loc["Slabý", "Na branku"]
    #     delta_g = g_weak - g_strong
    #     delta_s = d_weak - d_strong
    #     desc = "📌 Proti silným týmům tým skóruje méně" if delta_g > 0.3 else "📌 Výkon proti silným je vyrovnaný"
    #     desc += f", rozdíl v průměru gólů: **{delta_g:.2f}**, střel na branku: **{delta_s:.2f}**."
    #     st.markdown(desc)
