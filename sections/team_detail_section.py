import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.poisson_utils import (
    calculate_elo_ratings, calculate_form_emojis, calculate_expected_and_actual_points,
    aggregate_team_stats, calculate_team_pseudo_xg, add_btts_column,
    calculate_conceded_goals, calculate_recent_team_form,
    calculate_elo_changes, calculate_team_styles,
    calculate_clean_sheets, intensity_score_to_emoji, compute_score_stats, compute_form_trend,
    merged_home_away_opponent_form
)

def render_team_detail(df, season_df, team, league_name, gii_dict):
    st.sidebar.markdown("### ⏱️ Časový filtr")
    min_date, max_date = df['Date'].min(), df['Date'].max()
    time_filter = st.sidebar.radio("Vyber rozsah dat", ["Celá sezóna", "Posledních 5 zápasů", "Posledních 10 zápasů"])
    if time_filter == "Posledních 5 zápasů":
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values("Date", ascending=False).head(5)
        df = df[df['Date'].isin(team_matches['Date'])]
    elif time_filter == "Posledních 10 zápasů":
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values("Date", ascending=False).head(10)
        df = df[df['Date'].isin(team_matches['Date'])]

    df_sorted = df.sort_values("Date")
    df_sorted["DateDiff"] = df_sorted["Date"].diff().dt.days
    gap_threshold = 30
    latest_index = df_sorted[df_sorted["Date"] == df_sorted["Date"].max()].index[0]
    cutoff_idx = df_sorted.iloc[:latest_index].loc[df_sorted["DateDiff"] > gap_threshold].last_valid_index()
    season_start_date = df_sorted.loc[cutoff_idx + 1, "Date"] if cutoff_idx is not None else df_sorted["Date"].min()
    season_cutoff = df[df['Date'] >= season_start_date]  # nastavit začátek aktuální sezony podle datasetu
    if time_filter == "Posledních 5 zápasů":
        team_matches = season_cutoff[(season_cutoff['HomeTeam'] == team) | (season_cutoff['AwayTeam'] == team)].sort_values("Date", ascending=False).head(5)
        df = df[df['Date'].isin(team_matches['Date'])]
    elif time_filter == "Posledních 10 zápasů":
        team_matches = season_cutoff[(season_cutoff['HomeTeam'] == team) | (season_cutoff['AwayTeam'] == team)].sort_values("Date", ascending=False).head(10)
        df = df[df['Date'].isin(team_matches['Date'])]
    else:
        df = season_cutoff

    season_df = df

    st.header(f"📌 Detail týmu: {team}")

    team_stats = aggregate_team_stats(season_df)
    if team not in team_stats.index:
        st.error(f"Tým '{team}' nebyl nalezen v datech. Zkontroluj správnost názvu.")
        st.stop()

    from utils.poisson_utils import get_team_card_stats
    stats = team_stats.loc[team]
    card_stats = get_team_card_stats(season_df, team)
    stats['Žluté'] = card_stats['yellow']
    stats['Červené'] = card_stats['red']
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


    card_stats = get_team_card_stats(season_df, team)
    yellow_per_foul = card_stats["yellow"] / card_stats["fouls"] if card_stats["fouls"] else 0
    red_per_foul = card_stats["red"] / card_stats["fouls"] if card_stats["fouls"] else 0

    # ✅ Očekávané body
    xp_data = calculate_expected_and_actual_points(season_df).get(team, {})
    expected_points = xp_data.get("expected_points", 0)
    clean_pct = calculate_clean_sheets(season_df, team)

    # ✅ Přehledové metriky
    # První řádek metrik
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("⚽ Průměr gólů", f"{stats['Góly']:.2f}")
    col2.metric("🥅 Obdržené góly", f"{stats['Obdržené góly']:.2f}")
    col3.metric("📈 Intenzita", intensity_score_to_emoji(gii_dict.get(team, 0)))
    col4.metric("🎯 xG", f"{team_xg:.2f}")
    col5.metric("🛡️ xGA", f"{team_xga:.2f}")
    col6.metric("📊 Oček. body", f"{expected_points:.2f}", delta=f"Skutečné: {xp_data.get('points', 0):.2f}")

    # Druhý řádek průměrných statistik za sezónu
    col7, col8, col9, col10, col11, col12 = st.columns(6)
    col7.metric("📸 Průměr střel", f"{stats['Střely']:.2f}")
    col8.metric("🎯 Na branku", f"{stats['Na branku']:.2f}")
    col9.metric("🚩 Rohy", f"{stats['Rohy']:.2f}")
    col10.metric("⚠️ Fauly", f"{stats.get('HF', 0) + stats.get('AF', 0):.2f}")
    col11.metric("🟨 Žluté", f"{stats['Žluté']:.2f}")
    col12.metric("🟥 Červené", f"{stats['Červené']:.2f}")





    st.markdown("---")
    st.subheader("📊 Výkon doma vs venku – rozšířené statistiky")
    home = season_df[season_df['HomeTeam'] == team]
    away = season_df[season_df['AwayTeam'] == team]

    def calc_stats(df, prefix):
        return {
            f"{prefix} zápasy": len(df),
            f"{prefix} góly": df['FTHG'].mean() if 'FTHG' in df else 0,
            f"{prefix} střely": df['HS'].mean() if 'HS' in df else 0,
            f"{prefix} střely na branku": df['HST'].mean() if 'HST' in df else 0,
            f"{prefix} rohy": df['HC'].mean() if 'HC' in df else 0,
            f"{prefix} fauly": df['HF'].mean() if 'HF' in df else 0,
            f"{prefix} žluté karty": df['HY'].mean() if 'HY' in df else 0,
            f"{prefix} červené karty": df['HR'].mean() if 'HR' in df else 0
        }

    stats_home = calc_stats(home, "Doma")
    stats_away = calc_stats(away, "Venku")
    row_home = (
        pd.DataFrame(stats_home, index=["Hodnota"])
        .rename(columns={
            "Doma zápasy": "M", "Doma góly": "HG", "Doma střely": "HS",
            "Doma střely na branku": "HST", "Doma rohy": "HC", "Doma fauly": "HF",
            "Doma žluté karty": "HY", "Doma červené karty": "HR"
        })
        .round(1)
    )

    row_away = (
        pd.DataFrame(stats_away, index=["Hodnota"])
        .rename(columns={
            "Venku zápasy": "M", "Venku góly": "AG", "Venku střely": "AS",
            "Venku střely na branku": "AST", "Venku rohy": "AC", "Venku fauly": "AF",
            "Venku žluté karty": "AY", "Venku červené karty": "AR"
        })
        .round(1)
    )


    st.markdown("#### 🏠 Doma")
    st.table(row_home.style.format("{:.1f}"))

    st.markdown("#### 🚌 Venku")
    st.table(row_away.style.format("{:.1f}"))


    st.markdown("---")
    st.subheader("📅 Posledních 5 zápasů")

    last_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values("Date", ascending=False).head(5).copy()

    def format_result(row):
        location = "vs" if row['HomeTeam'] == team else "@"
        opponent = row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam']
        team_goals = row['FTHG'] if row['HomeTeam'] == team else row['FTAG']
        opp_goals = row['FTAG'] if row['HomeTeam'] == team else row['FTHG']
        return pd.Series({
            "Datum": row['Date'].date(),
            "Soupeř": f"{location} {opponent}",
            "Skóre": f"{team_goals}:{opp_goals}",
            "Střely": row['HS'] if row['HomeTeam'] == team else row['AS'],
            "Na branku": row['HST'] if row['HomeTeam'] == team else row['AST'],
            "Rohy": row['HC'] if row['HomeTeam'] == team else row['AC'],
            "Fauly": row['HF'] if row['HomeTeam'] == team else row['AF'],
            "Žluté": row['HY'] if row['HomeTeam'] == team else row['AY'],
            "Červené": row['HR'] if row['HomeTeam'] == team else row['AR']
        })

    match_details = last_matches.apply(format_result, axis=1)

    def highlight_result(row):
        score = row["Skóre"].split(":")
        if len(score) != 2:
            return [""] * len(row)
        team_goals, opp_goals = int(score[0]), int(score[1])
        color = "background-color: #d4edda" if team_goals > opp_goals else ("background-color: #f8d7da" if team_goals < opp_goals else "background-color: #fff3cd")
        return [color] * len(row)

    styled_matches = match_details.style.apply(highlight_result, axis=1)
    st.table(styled_matches.hide(axis="index").set_table_attributes('style="width: 100%;"').set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]))

    st.markdown("---")
    st.subheader("🔍 Doplňkové výkonnostní statistiky")

    # Ligový průměr
    league_stats = team_stats.mean()
    def compare_stat(name, team_value):
        league_avg = league_stats.get(name, 0)
        diff = team_value - league_avg
        return f" *(liga: {league_avg:.2f}, Δ {diff:+.2f})*"

    # Disciplinovanost – karty na faul
    total_fouls = stats.get('HF', 0) + stats.get('AF', 0)
    st.markdown(f"- 🟨 Žluté na faul: **{yellow_per_foul:.2f}**")
    st.markdown(f"- 🟥 Červené na faul: **{red_per_foul:.2f}**")
    # Ofenzivní efektivita – střely na 1 gól
    offensive_efficiency = (stats['Střely'] / stats['Góly']) if stats['Góly'] else 0
    st.markdown(f"- ⚡ Ofenzivní efektivita (střel na gól): **{offensive_efficiency:.2f}**" + compare_stat("Střely", stats['Střely']))
    defensive_efficiency = (stats['Obdržené góly'] / stats['Střely']) if stats['Střely'] else 0
    st.markdown(f"- 🛡️ Defenzivní efektivita (góly na střelu): **{defensive_efficiency:.2f}**" + compare_stat("Obdržené góly", stats['Obdržené góly']))
    avg_goals = stats['Góly']
    avg_conceded = stats['Obdržené góly']
    conversion_rate = (avg_goals / stats['Střely']) if stats['Střely'] else 0
    shot_accuracy = (stats['Na branku'] / stats['Střely']) if stats['Střely'] else 0
    st.markdown(f"- 🎯 Přesnost střel: **{shot_accuracy * 100:.1f}%**" + compare_stat("Na branku", stats['Na branku']))
    st.markdown(f"- 💥 Konverzní míra: **{conversion_rate * 100:.1f}%**" + compare_stat("Góly", stats['Góly']))
    total_matches = len(season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)])
    wins = season_df[((season_df['HomeTeam'] == team) & (season_df['FTR'] == 'H')) | ((season_df['AwayTeam'] == team) & (season_df['FTR'] == 'A'))]
    draws = season_df[season_df['FTR'] == 'D'][(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)]
    losses = total_matches - len(wins) - len(draws)
    over25 = season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)]
    over25_ratio = (over25['FTHG'] + over25['FTAG'] > 2.5).mean() * 100
    btts_ratio = ((over25['FTHG'] > 0) & (over25['FTAG'] > 0)).mean() * 100
    st.markdown(f"- ✅ Výher: **{len(wins)}**  | 🤝 Remíz: **{len(draws)}** | ❌ Proher: **{losses}**")
    st.markdown(f"- 🔼 Zápasy s více než 2.5 góly: **{over25_ratio:.1f}%**")
    st.markdown(f"- 🎯 Zápasy, kde skórovaly oba týmy (BTTS): **{btts_ratio:.1f}%**")

    st.markdown("---")
    from utils.poisson_utils import classify_team_strength

    df_team = season_df[(season_df['HomeTeam'] == team) | (season_df['AwayTeam'] == team)].copy()
    df_team['Opponent'] = df_team.apply(lambda row: row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam'], axis=1)
    df_team['Strength'] = df_team['Opponent'].apply(lambda t: classify_team_strength(season_df, t))

    def extract_match_stats(row):
        is_home = row['HomeTeam'] == team
        return pd.Series({
            "Soupeř": row['Opponent'],
            "Typ soupeře": row['Strength'],
            "Góly": row['FTHG'] if is_home else row['FTAG'],
            "Střely": row['HS'] if is_home else row['AS'],
            "Na branku": row['HST'] if is_home else row['AST'],
            "Rohy": row['HC'] if is_home else row['AC'],
            "Fauly": row['HF'] if is_home else row['AF'],
            "Žluté": row['HY'] if is_home else row['AY'],
            "Červené": row['HR'] if is_home else row['AR']
        })

    detailed_stats = df_team.apply(extract_match_stats, axis=1)
    numeric_columns = detailed_stats.select_dtypes(include='number').columns
    avg_by_strength = detailed_stats.groupby("Typ soupeře")[numeric_columns].mean().round(2)
    st.markdown("---")
    st.subheader("📉 Výkonnost proti kategoriím soupeřů")
    st.markdown("Souhrnné statistiky proti různě silným soupeřům")

    # Shrnutí disciplíny
    if total_fouls > 0:
        if yellow_per_foul > 0.25:
            st.markdown(f"🟡 Tým fauluje poměrně neukázněně – **{yellow_per_foul:.2f}** žlutých na 1 faul.")
        else:
            st.markdown(f"🟢 Tým je relativně disciplinovaný – **{yellow_per_foul:.2f}** žlutých na 1 faul.")

        if red_per_foul > 0.05:
            st.markdown(f"🔴 Relativně vysoký výskyt červených karet: **{red_per_foul:.2f}** na faul.")

    # Shrnutí konverze a obrany
    # Souhrnný závěr
    if conversion_rate > 0.15 and defensive_efficiency > 0.12:
        st.markdown("🔁 **Souhrn:** Tým má ofenzivní sílu, ale defenzivní slabiny.")
    elif conversion_rate < 0.08 and defensive_efficiency < 0.07:
        st.markdown("🧤 **Souhrn:** Tým je defenzivně pevný, ale v útoku neefektivní.")
    elif conversion_rate > 0.15 and defensive_efficiency < 0.07:
        st.markdown("💪 **Souhrn:** Tým je dominantní na obou stranách – silný útok i obrana.")
    elif conversion_rate < 0.08 and defensive_efficiency > 0.12:
        st.markdown("⚠️ **Souhrn:** Tým má potíže v útoku i v obraně.")
    if conversion_rate > 0.15:
        st.markdown(f"⚽ Tým má vysokou konverzní míru – **{conversion_rate*100:.1f}%** střel končí gólem.")
    elif conversion_rate < 0.08:
        st.markdown(f"🚫 Nízká konverzní míra – pouze **{conversion_rate*100:.1f}%** střel je gólových.")

    if defensive_efficiency > 0.12:
        st.markdown(f"❗ Tým dostává gól z každé 8. střely – defenziva je zranitelná.")
    st.table(avg_by_strength.style.format("{:.2f}"))

    # Verbální shrnutí výkonu proti kategoriím soupeřů
    if set(["Silný", "Slabý"]).issubset(avg_by_strength.index):
        g_strong = avg_by_strength.loc["Silný", "Góly"]
        g_weak = avg_by_strength.loc["Slabý", "Góly"]
        d_strong = avg_by_strength.loc["Silný", "Na branku"]
        d_weak = avg_by_strength.loc["Slabý", "Na branku"]
        delta_g = g_weak - g_strong
        delta_s = d_weak - d_strong
        desc = "📌 Proti silným týmům tým skóruje méně" if delta_g > 0.3 else "📌 Výkon proti silným je vyrovnaný"
        desc += f", rozdíl v průměru gólů: **{delta_g:.2f}**, střel na branku: **{delta_s:.2f}**."
        st.markdown(desc)
