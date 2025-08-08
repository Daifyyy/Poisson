import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from utils.poisson_utils import (
    calculate_elo_ratings, calculate_form_emojis, calculate_expected_and_actual_points,
    aggregate_team_stats, calculate_team_pseudo_xg, add_btts_column,
    calculate_conceded_goals, calculate_recent_team_form,
    calculate_elo_changes, calculate_team_styles,
    calculate_clean_sheets, intensity_score_to_emoji, compute_score_stats, compute_form_trend,
    merged_home_away_opponent_form,classify_team_strength,calculate_advanced_team_metrics,
    calculate_team_extra_stats,get_team_record,analyze_team_profile,generate_team_comparison,render_team_comparison_section
)

def render_team_detail(
    df: pd.DataFrame,
    season_df: pd.DataFrame,
    team: str,
    league_name: str,
    gii_dict: Dict[str, float],
) -> None:
    """Render detailed information for a selected team.

    Args:
        df: DataFrame containing match data.
        season_df: DataFrame with season data for statistics.
        team: Team name to display.
        league_name: Name of the league.
        gii_dict: Mapping of teams to intensity index values.

    Returns:
        None
    """
    st.sidebar.markdown("### ⏱️ Časový filtr")
    min_date, max_date = df['Date'].min(), df['Date'].max()

    time_filter = st.sidebar.radio(
        "Vyber rozsah dat",
        ["Celá sezóna", "Posledních 5 zápasů", "Posledních 10 zápasů", "Posledních 5 doma", "Posledních 5 venku"]
    )

    def _apply_time_filter(data: pd.DataFrame) -> pd.DataFrame:
        df_sorted = data.sort_values("Date")
        df_sorted["DateDiff"] = df_sorted["Date"].diff().dt.days
        gap_threshold = 30
        latest_index = df_sorted[df_sorted["Date"] == df_sorted["Date"].max()].index[0]
        cutoff_idx = df_sorted.iloc[:latest_index].loc[df_sorted["DateDiff"] > gap_threshold].last_valid_index()
        season_start = df_sorted.loc[cutoff_idx + 1, "Date"] if cutoff_idx is not None else df_sorted["Date"].min()
        season_cutoff = data[data['Date'] >= season_start]

        if time_filter == "Posledních 5 zápasů":
            matches = season_cutoff[(season_cutoff['HomeTeam'] == team) | (season_cutoff['AwayTeam'] == team)]
            return matches.sort_values("Date", ascending=False).head(5)
        if time_filter == "Posledních 10 zápasů":
            matches = season_cutoff[(season_cutoff['HomeTeam'] == team) | (season_cutoff['AwayTeam'] == team)]
            return matches.sort_values("Date", ascending=False).head(10)
        if time_filter == "Posledních 5 doma":
            matches = season_cutoff[season_cutoff['HomeTeam'] == team]
            return matches.sort_values("Date", ascending=False).head(5)
        if time_filter == "Posledních 5 venku":
            matches = season_cutoff[season_cutoff['AwayTeam'] == team]
            return matches.sort_values("Date", ascending=False).head(5)
        return season_cutoff

    df = _apply_time_filter(df)

    difficulty_filter = st.sidebar.selectbox(
        "🎯 Filtrovat podle síly soupeře:",
        ["Vše", "Silní", "Průměrní", "Slabí"]
    )

    compare_team = st.sidebar.selectbox(
        "🔄 Porovnat s jiným týmem:",
        ["Žádný"] + sorted(df['HomeTeam'].unique().tolist())
    )

    def _apply_difficulty_filter(data: pd.DataFrame) -> pd.DataFrame:
        if difficulty_filter != "Vše":
            data = data.copy()
            data["Opponent"] = data.apply(lambda row: row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam'], axis=1)
            data["Soupeř síla"] = data["Opponent"].apply(lambda opp: classify_team_strength(df, opp))
            data = data[data["Soupeř síla"] == difficulty_filter]
        return data

    season_df = _apply_difficulty_filter(df)

    home = season_df[season_df['HomeTeam'] == team]
    away = season_df[season_df['AwayTeam'] == team]
    all_matches = pd.concat([home, away])

    if compare_team and compare_team != "Žádný" and compare_team != team:
        stats_all = calculate_advanced_team_metrics(all_matches)
        stats_home = calculate_advanced_team_metrics(home, is_home=True)
        stats_away = calculate_advanced_team_metrics(away, is_home=False)

        # ⬇️ použij původní df (ne season_df) pro druhý tým
        compare_home = df[df['HomeTeam'] == compare_team]
        compare_away = df[df['AwayTeam'] == compare_team]

        compare_matches = pd.concat([compare_home, compare_away])

        if team in stats_all.index and compare_team in calculate_advanced_team_metrics(compare_matches).index:
            render_team_comparison_section(
                team, compare_team,
                stats_all.loc[team], stats_home.loc[team], stats_away.loc[team],
                calculate_advanced_team_metrics(compare_matches).loc[compare_team],
                calculate_advanced_team_metrics(compare_home, is_home=True).loc[compare_team],
                calculate_advanced_team_metrics(compare_away, is_home=False).loc[compare_team]
            )
            return
        else:
            st.warning("⚠️ Jeden z týmů nemá dostupná data pro zvolený filtr.")
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



    
    team_stats = aggregate_team_stats(season_df)
    if team not in team_stats.index:
        st.error(f"Tým '{team}' nebyl nalezen v datech. Zkontroluj správnost názvu.")
        st.stop()

    
    # Ligový průměr
    league_avg = team_stats.mean()
    def compare_stat(name, team_value, league_avg):
        league_value = league_avg.get(name, 0)
        diff = team_value - league_value
        return f" *(liga: {league_value:.2f}, Δ {diff:+.2f})*"

    advanced_stats = calculate_advanced_team_metrics(season_df)
    league_avg_advanced = advanced_stats.mean()

    
    
    
    stats = team_stats.loc[team]
    #card_stats = get_team_card_stats(season_df, team)
    # stats['Žluté'] = card_stats['yellow']
    # stats['Červené'] = card_stats['red']
    elo_dict = calculate_elo_ratings(season_df)

    # ✅ Kontrola rozsahu dat a počtu zápasů
    st.caption(f"Počet zápasů v aktuálním datasetu: {len(season_df)}")
    st.caption(f"Rozsah dat: {season_df['Date'].min().date()} až {season_df['Date'].max().date()}")

    # ✅ xG a xGA – kontrola struktury + fallback
    # Výpočet xG / xGA se zarovnáním názvu týmu
    from utils.poisson_utils.xg import calculate_team_pseudo_xg
    xg_dict = calculate_team_pseudo_xg(season_df)
    # klíče bez mezer a lowercase (normalize)
    normalized_xg_dict = {k.strip().lower(): v for k, v in xg_dict.items()}
    normalized_team = team.strip().lower()

    team_xg_data = normalized_xg_dict.get(normalized_team, {})
    team_xg = team_xg_data.get("xg", 0)
    team_xga = team_xg_data.get("xga", 0)

    # Výpočet BTTS, Over 2.5 a průměr celkových gólů
    team_matches = season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)].copy()
    team_matches = team_matches.dropna(subset=["FTHG", "FTAG"])

    btts_pct = 100 * ((team_matches["FTHG"] > 0) & (team_matches["FTAG"] > 0)).mean()
    over25_pct = 100 * ((team_matches["FTHG"] + team_matches["FTAG"]) > 2.5).mean()
    avg_total_goals = (team_matches["FTHG"] + team_matches["FTAG"]).mean()

    #card_stats = get_team_card_stats(season_df, team)
    # yellow_per_foul = card_stats["yellow"] / card_stats["fouls"] if card_stats["fouls"] else 0
    # red_per_foul = card_stats["red"] / card_stats["fouls"] if card_stats["fouls"] else 0

    # ✅ Očekávané body
    xp_data = calculate_expected_and_actual_points(season_df).get(team, {})
    expected_points = xp_data.get("expected_points", 0)
    clean_pct = calculate_clean_sheets(season_df, team)

    

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
    col_all, col_home, col_away = st.columns(3)

    # Funkce pro delta barvy
    def colored_delta(value, league_avg, metric_name):
        diff = value - league_avg
        arrow = "⬆️" if diff > 0 else "⬇️"
        
        # Metriky, kde vyšší hodnota je špatně
        inverse_metrics = ["Obdržené góly", "Fauly", "Žluté", "Červené"]
        inverse = metric_name in inverse_metrics
        
        color = "red" if (diff > 0 and inverse) or (diff < 0 and not inverse) else "green"
        return f"<span style='color:{color}'>{arrow} {diff:+.2f}</span>"

    # Funkce pro výpis jednoho sloupce
    # def display_metrics_block(col, title, data, adv_data, extra):
    #     with col:
    #         st.markdown(f"#### {title}")
            
    #         # Základní metriky
    #         st.markdown(f"**⚽ Góly:** {data['Góly']:.2f} {colored_delta(data['Góly'], league_avg['Góly'], 'Góly')}", unsafe_allow_html=True)
    #         st.markdown(f"**🥅 Obdržené góly:** {data['Obdržené góly']:.2f} {colored_delta(data['Obdržené góly'], league_avg['Obdržené góly'], 'Obdržené góly')}", unsafe_allow_html=True)
    #         st.markdown(f"**📸 Střely:** {data['Střely']:.2f} {colored_delta(data['Střely'], league_avg['Střely'], 'Střely')}", unsafe_allow_html=True)
    #         st.markdown(f"**🎯 Na branku:** {data['Na branku']:.2f} {colored_delta(data['Na branku'], league_avg['Na branku'], 'Na branku')}", unsafe_allow_html=True)
    #         st.markdown(f"**🚩 Rohy:** {data['Rohy']:.2f} {colored_delta(data['Rohy'], league_avg['Rohy'], 'Rohy')}", unsafe_allow_html=True)
    #         st.markdown(f"**⚠️ Fauly:** {data['Fauly']:.2f} {colored_delta(data['Fauly'], league_avg['Fauly'], 'Fauly')}", unsafe_allow_html=True)
    #         st.markdown(f"**🟨 Žluté:** {data['Žluté']:.2f} {colored_delta(data['Žluté'], league_avg['Žluté'], 'Žluté')}", unsafe_allow_html=True)
    #         st.markdown(f"**🟥 Červené:** {data['Červené']:.2f} {colored_delta(data['Červené'], league_avg['Červené'], 'Červené')}", unsafe_allow_html=True)
    #         # st.markdown(f"**⚡ Ofenzivní efektivita:** {adv_data['Ofenzivní efektivita']:.2f} {colored_delta(adv_data['Ofenzivní efektivita'], league_avg_advanced['Ofenzivní efektivita'], 'Ofenzivní efektivita')}", unsafe_allow_html=True)
    #         # st.markdown(f"**🛡️ Defenzivní efektivita:** {adv_data['Defenzivní efektivita']:.2f} {colored_delta(adv_data['Defenzivní efektivita'], league_avg_advanced['Defenzivní efektivita'], 'Defenzivní efektivita')}", unsafe_allow_html=True)
    #         st.markdown(f"**🎯 Přesnost střel:** {adv_data['Přesnost střel'] * 100:.1f}% {colored_delta(adv_data['Přesnost střel'], league_avg_advanced['Přesnost střel'], 'Přesnost střel')}", unsafe_allow_html=True)
    #         st.markdown(f"**🌟 Konverzní míra:** {adv_data['Konverzní míra'] * 100:.1f}% {colored_delta(adv_data['Konverzní míra'], league_avg_advanced['Konverzní míra'], 'Konverzní míra')}", unsafe_allow_html=True)
    #         st.markdown(f"**🧤 Čistá konta:** {extra['Čistá konta %']:.1f}%", unsafe_allow_html=True)
    #         st.markdown(f"**📈 Over 2.5 %:** {extra['Over 2.5 %']:.1f}%", unsafe_allow_html=True)
    #         st.markdown(f"**🎯 BTTS %:** {extra['BTTS %']:.1f}%", unsafe_allow_html=True)

    def display_metrics_block(col, title, data, advanced, extra, show_labels=True):
        with col:
            st.markdown(f"### {title}")

            def format_metric(label, value, delta_str):
                if show_labels:
                    return f"**{label}:** {value:.2f} {delta_str}"
                else:
                    return f"{value:.2f} {delta_str}"

            st.markdown(format_metric("⚽ Góly", data['Góly'], colored_delta(data['Góly'], league_avg['Góly'], 'Góly')), unsafe_allow_html=True)
            st.markdown(format_metric("🥅 Obdržené góly", data['Obdržené góly'], colored_delta(data['Obdržené góly'], league_avg['Obdržené góly'], 'Obdržené góly')), unsafe_allow_html=True)
            st.markdown(format_metric("📸 Střely", data['Střely'], colored_delta(data['Střely'], league_avg['Střely'], 'Střely')), unsafe_allow_html=True)
            st.markdown(format_metric("🎯 Na branku", data['Na branku'], colored_delta(data['Na branku'], league_avg['Na branku'], 'Na branku')), unsafe_allow_html=True)
            st.markdown(format_metric("🚩 Rohy", data['Rohy'], colored_delta(data['Rohy'], league_avg['Rohy'], 'Rohy')), unsafe_allow_html=True)
            st.markdown(format_metric("⚠️ Fauly", data['Fauly'], colored_delta(data['Fauly'], league_avg['Fauly'], 'Fauly')), unsafe_allow_html=True)
            st.markdown(format_metric("🟨 Žluté", data['Žluté'], colored_delta(data['Žluté'], league_avg['Žluté'], 'Žluté')), unsafe_allow_html=True)
            st.markdown(format_metric("🟥 Červené", data['Červené'], colored_delta(data['Červené'], league_avg['Červené'], 'Červené')), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(format_metric("🎯 Přesnost střel", advanced["Přesnost střel"], colored_delta(advanced["Přesnost střel"], league_avg_advanced["Přesnost střel"], "Přesnost střel")), unsafe_allow_html=True)
            st.markdown(format_metric("🌟 Konverzní míra", advanced["Konverzní míra"], colored_delta(advanced["Konverzní míra"], league_avg_advanced["Konverzní míra"], "Konverzní míra")), unsafe_allow_html=True)
            st.markdown(format_metric("🧤 Čistá konta", extra["Čistá konta %"], ""), unsafe_allow_html=True)
            st.markdown(format_metric("📈 Over 2.5 %", extra["Over 2.5 %"], ""), unsafe_allow_html=True)
            st.markdown(format_metric("🎯 BTTS %", extra["BTTS %"], ""), unsafe_allow_html=True)


    
    # Celkem
    # display_metrics_block(col_all, "Celkem", metrics_all, advanced_stats.loc[team], extra_all)

    # # Doma
    # home_adv = calculate_advanced_team_metrics(home)
    # if not home_adv.empty and team in home_adv.index:
    #     display_metrics_block(col_home, "🏠 Doma", metrics_home, home_adv.loc[team], extra_home)

    # # Venku
    # away_adv = calculate_advanced_team_metrics(away)
    # if not away_adv.empty and team in away_adv.index:
    #     display_metrics_block(col_away, "🚌 Venku", metrics_away, away_adv.loc[team], extra_away)

    # Celkem
    display_metrics_block(col_all, "Celkem", metrics_all, advanced_stats.loc[team], extra_all, show_labels=True)

    # Doma
    home_adv = calculate_advanced_team_metrics(home)
    if not home_adv.empty and team in home_adv.index:
        display_metrics_block(col_home, "🏠 Doma", metrics_home, home_adv.loc[team], extra_home, show_labels=False)

    # Venku
    away_adv = calculate_advanced_team_metrics(away)
    if not away_adv.empty and team in away_adv.index:
        display_metrics_block(col_away, "🚌 Venku", metrics_away, away_adv.loc[team], extra_away, show_labels=False)


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
            "Rohy": row['HC'] if is_home else row['AC'],
            "Fauly": row['HF'] if is_home else row['AF'],
            "Žluté": row['HY'] if is_home else row['AY'],
            "Červené": row['HR'] if is_home else row['AR'],
        })

    # ✅ Převod a styling
    match_details = last_matches.apply(format_result, axis=1)
    match_details = match_details.reset_index(drop=True)  # ✅ odstraní indexový sloupec

    def highlight_result(row):
        score = row["Skóre"].split(":")
        if len(score) != 2:
            return [""] * len(row)
        team_goals, opp_goals = int(score[0]), int(score[1])
        color = "#d4edda" if team_goals > opp_goals else "#f8d7da" if team_goals < opp_goals else "#fff3cd"
        return [f"background-color: {color}"] * len(row)

    styled_matches = match_details.style.apply(highlight_result, axis=1)

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

    
    # def extract_match_stats(row):
    #     is_home = row['HomeTeam'] == team
    #     return pd.Series({
    #         "Soupeř": row['Opponent'],
    #         "Typ soupeře": row['Strength'],
    #         "Góly": row['FTHG'] if is_home else row['FTAG'],
    #         "Střely": row['HS'] if is_home else row['AS'],
    #         "Na branku": row['HST'] if is_home else row['AST'],
    #         "Rohy": row['HC'] if is_home else row['AC'],
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
