import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

import plotly.graph_objects as go

from utils.responsive import responsive_columns
from utils.poisson_utils import (
    elo_history, calculate_form_emojis, calculate_expected_and_actual_points,
    aggregate_team_stats, detect_current_season,
    get_team_xg_xga, calculate_conceded_goals, calculate_recent_team_form,
    calculate_elo_changes, calculate_team_styles,
    intensity_score_to_emoji, compute_score_stats, compute_form_trend,
    merged_home_away_opponent_form, classify_team_strength, calculate_advanced_team_metrics,
    calculate_team_extra_stats, get_team_record, analyze_team_profile, generate_team_comparison,
    calculate_mdi, render_team_comparison_section
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
    """Render detailed information for a selected team."""
    st.sidebar.markdown("### ⏱️ Časový filtr")

    time_filter = st.sidebar.radio(
        "Vyber rozsah dat",
        ["Celá sezóna", "Posledních 5 zápasů", "Posledních 10 zápasů", "Posledních 5 doma", "Posledních 5 venku"]
    )

    def _apply_time_filter(data: pd.DataFrame, team_name: str):
        df_sorted = data.sort_values("Date")
        df_sorted["DateDiff"] = df_sorted["Date"].diff().dt.days
        gap_threshold = 30
        latest_index = df_sorted[df_sorted["Date"] == df_sorted["Date"].max()].index[0]
        cutoff_idx = df_sorted.iloc[:latest_index].loc[df_sorted["DateDiff"] > gap_threshold].last_valid_index()
        season_start = df_sorted.loc[cutoff_idx + 1, "Date"] if cutoff_idx is not None else df_sorted["Date"].min()
        season_cutoff = data[data['Date'] >= season_start]

        base_matches = season_cutoff[(season_cutoff['HomeTeam'] == team_name) | (season_cutoff['AwayTeam'] == team_name)]
        recent_all = base_matches.sort_values("Date", ascending=False).head(5)
        recent_home = season_cutoff[season_cutoff['HomeTeam'] == team_name].sort_values("Date", ascending=False).head(5)
        recent_away = season_cutoff[season_cutoff['AwayTeam'] == team_name].sort_values("Date", ascending=False).head(5)

        if time_filter == "Posledních 5 zápasů":
            selected = recent_all
        elif time_filter == "Posledních 10 zápasů":
            selected = base_matches.sort_values("Date", ascending=False).head(10)
        elif time_filter == "Posledních 5 doma":
            selected = recent_home
        elif time_filter == "Posledních 5 venku":
            selected = recent_away
        else:
            selected = season_cutoff

        return selected, {"all": recent_all, "home": recent_home, "away": recent_away}

    original_df = season_df
    filtered_df, recent_matches = _apply_time_filter(original_df, team)

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

    recent_all = _apply_difficulty_filter(recent_matches["all"])
    recent_home = _apply_difficulty_filter(recent_matches["home"])
    recent_away = _apply_difficulty_filter(recent_matches["away"])

    home = season_df[season_df['HomeTeam'] == team]
    away = season_df[season_df['AwayTeam'] == team]
    all_matches = pd.concat([home, away])

    season_start = detect_current_season(original_df, prepared=True)[1]
    season = str(season_start.year)

    if compare_team and compare_team != "Žádný" and compare_team != team:
        compare_df, _ = _apply_time_filter(original_df, compare_team)
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

    st.header(f"📌 Detail týmu: {team}")

    # Výpočty
    record_all = get_team_record(season_df, team)
    record_home = get_team_record(season_df, team, side="home")
    record_away = get_team_record(season_df, team, side="away")

    st.markdown(
        f"**📊 Bilance:** &nbsp;&nbsp;&nbsp;"
        f"🟦 Celkem – ✅ {record_all[0]} | 🤝 {record_all[1]} | ❌ {record_all[2]} &nbsp;&nbsp;&nbsp;"
        f"🏠 Doma – ✅ {record_home[0]} | 🤝 {record_home[1]} | ❌ {record_home[2]} &nbsp;&nbsp;&nbsp;"
        f"🚌 Venku – ✅ {record_away[0]} | 🤝 {record_away[1]} | ❌ {record_away[2]}",
        unsafe_allow_html=True
    )

    ws_stats = get_team_xg_xga(team, season, season_df)
    team_xg = ws_stats.get("xg", np.nan)
    team_xga = ws_stats.get("xga", np.nan)

    col_xg, col_xga = st.columns(2)
    col_xg.metric("Sezónní xG", f"{team_xg:.1f}")
    col_xga.metric("Sezónní xGA", f"{team_xga:.1f}")

    team_stats = aggregate_team_stats(season_df)
    if team not in team_stats.index:
        st.error(f"Tým '{team}' nebyl nalezen v datech. Zkontroluj správnost názvu.")
        st.stop()

    league_avg = team_stats.mean()

    advanced_stats = calculate_advanced_team_metrics(season_df)
    league_avg_advanced = advanced_stats.mean()

    stats = team_stats.loc[team]

    st.caption(f"Počet zápasů v aktuálním datasetu: {len(season_df)}")
    st.caption(f"Rozsah dat: {season_df['Date'].min().date()} až {season_df['Date'].max().date()}")

    home_xg = away_xg = team_xg
    home_xga = away_xga = team_xga

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
    df_team['Opponent'] = df_team.apply(lambda row: row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam'], axis=1)
    df_team['H/A'] = df_team.apply(lambda row: 'H' if row['HomeTeam'] == team else 'A', axis=1)
    df_team['Soupeř síla'] = df_team['Opponent'].apply(lambda opp: classify_team_strength(season_df, opp))

    if difficulty_filter != "Vše":
        df_team = df_team[df_team["Soupeř síla"] == difficulty_filter]

    df_team = df_team.dropna(subset=["FTHG", "FTAG"])
    last_matches = df_team.sort_values("Date", ascending=False).head(5)

    def format_result(row):
        is_home = row['HomeTeam'] == team
        opponent = row['AwayTeam'] if is_home else row['HomeTeam']
        team_goals = row['FTHG'] if is_home else row['FTAG']
        opp_goals = row['FTAG'] if is_home else row['FTHG']
        return pd.Series({
            "Datum": row['Date'].date(),
            "Soupeř": opponent,
            "H/A": "H" if is_home else "A",
            "Skóre": f"{team_goals}:{opp_goals}",
            "Střely": row['HS'] if is_home else row['AS'],
            "Na branku": row['HST'] if is_home else row['AST'],
            "Fauly": row['HF'] if is_home else row['AF'],
            "Žluté": row['HY'] if is_home else row['AY'],
            "Červené": row['HR'] if is_home else row['AR'],
        })

    match_details = last_matches.apply(format_result, axis=1).reset_index(drop=True)

    def highlight_result(row):
        score = row["Skóre"].split(":")
        if len(score) != 2 or not all(part.isdigit() for part in score):
            return [""] * len(row)
        team_goals, opp_goals = map(int, score)
        color = "#d4edda" if team_goals > opp_goals else "#f8d7da" if team_goals < opp_goals else "#fff3cd"
        return [f"background-color: {color}"] * len(row)

    styled_matches = match_details.style.apply(highlight_result, axis=1).format(precision=1)

    st.markdown("### 🕵️ Posledních 5 zápasů")
    st.table(styled_matches)

    st.subheader("📊 Match Dominance Index (MDI)")

    # Ligové průměry přes časově filtrovaná data
    league_avgs = filtered_df[
        ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]
    ].mean().to_dict()

    strength_map = {"Silní": 1.1, "Průměrní": 1.0, "Slabí": 0.9}

    def build_mdi_df(df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            is_home = row['HomeTeam'] == team
            opponent = row['AwayTeam'] if is_home else row['HomeTeam']
            strength_label = classify_team_strength(season_df, opponent)
            coeff = strength_map.get(strength_label, 1.0)

            # Přepočet statistik na perspektivu zvoleného týmu
            row_for_mdi = row.copy()
            if not is_home:
                swap_cols = {
                    "HS": "AS", "AS": "HS",
                    "HST": "AST", "AST": "HST",
                    "HC": "AC", "AC": "HC",
                    "HF": "AF", "AF": "HF",
                    "HY": "AY", "AY": "HY",
                    "HR": "AR", "AR": "HR",
                }
                row_for_mdi = row_for_mdi.rename(swap_cols)

            mdi_val = calculate_mdi(row_for_mdi, league_avgs, coeff)

            team_goals = row['FTHG'] if is_home else row['FTAG']
            opp_goals = row['FTAG'] if is_home else row['FTHG']
            records.append(
                {
                    "Datum": row['Date'].date(),
                    "Soupeř": opponent,
                    "Výsledek": f"{team_goals}-{opp_goals}",
                    "MDI": mdi_val,
                }
            )
        return pd.DataFrame(records)

    mdi_all = build_mdi_df(recent_all)
    mdi_home = build_mdi_df(recent_home)
    mdi_away = build_mdi_df(recent_away)

    mdi_option = st.radio("MDI filtr", ["Posledních 5", "Posledních 5 doma", "Posledních 5 venku"])
    mdi_df = {
        "Posledních 5": mdi_all,
        "Posledních 5 doma": mdi_home,
        "Posledních 5 venku": mdi_away,
    }[mdi_option]

    if not mdi_df.empty:
        mdi_df["Soupeř a výsledek"] = mdi_df["Soupeř"] + " " + mdi_df["Výsledek"]
        fig_mdi = go.Figure()
        fig_mdi.add_trace(
            go.Bar(
                x=mdi_df["MDI"],
                y=mdi_df["Soupeř a výsledek"],
                customdata=mdi_df["Datum"],
                orientation="h",
                hovertemplate="%{y}<br>Datum: %{customdata}<br>MDI: %{x:.1f}<extra></extra>",
            )
        )
        # Referenční čára na MDI=50
        fig_mdi.add_vline(x=50, line_dash="dash", line_color="red")
        fig_mdi.update_layout(
            xaxis_title="MDI",
            yaxis_title="Soupeř",
            xaxis=dict(range=[0, 100]),
            showlegend=False,
        )
        st.plotly_chart(fig_mdi, use_container_width=True)
    else:
        st.info("MDI není dostupné pro zvolený filtr")

    # Disciplinovanost – karty na faul
    yellow_per_foul = stats['Žluté'] / stats['Fauly'] if stats['Fauly'] else 0
    red_per_foul = stats.get('Červené', 0) / stats['Fauly'] if stats['Fauly'] else 0

    defensive_efficiency = (stats['Obdržené góly'] / stats['Střely']) if stats['Střely'] else 0
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
