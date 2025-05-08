import pandas as pd
import numpy as np

from ..utils_warnings import classify_team_strength  # pokud není globálně importováno
from .core import prepare_df


def calculate_match_tempo(df: pd.DataFrame, team: str, opponent_elo: float, is_home: bool, elo_dict: dict, last_n: int = 10) -> dict:
    """Spočítá tempo, dominanci a tvrdost zápasu na základě posledních zápasů proti podobně silným soupeřům (ELO + typ soupeře)."""
    from datetime import timedelta
    import numpy as np

    df = prepare_df(df)
    latest_date = df['Date'].max()
    one_year_ago = latest_date - timedelta(days=365)
    df = df[df['Date'] >= one_year_ago]  # ✅ filtrujeme poslední rok

    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    opp_col = 'AwayTeam' if is_home else 'HomeTeam'

    # Všechny zápasy daného týmu
    matches_all = df[df[team_col] == team].copy()
    matches_all['OppELO'] = matches_all[opp_col].map(elo_dict)
    matches_all['EloDiff'] = abs(matches_all['OppELO'] - opponent_elo)

    # ✅ postupné rozšiřování ELO rozdílu
    max_elo_diff = 50
    step = 25
    matches = matches_all[matches_all['EloDiff'] <= max_elo_diff]
    while len(matches) < last_n and max_elo_diff <= 200:
        max_elo_diff += step
        matches = matches_all[matches_all['EloDiff'] <= max_elo_diff]

    if matches.empty:
        print(f"[{team}] ⚠️ Nenalezeny žádné relevantní zápasy — vracím defaultní nuly.")
        return {
            "tempo": 0,
            "percentile": 0,
            "rating": "N/A",
            "imbalance": 0.0,
            "imbalance_type": "N/A",
            "aggressiveness_index": 0.0,
            "aggressiveness_rating": "N/A",
            "similar_opponents_tempo": 0.0
        }

    print(f"[{team}] ✅ Použito {len(matches)} zápasů (ELO diff ≤ {max_elo_diff})")

    # Tempo
    shots = matches['HS'] if is_home else matches['AS']
    corners = matches['HC'] if is_home else matches['AC']
    fouls = matches['HF'] if is_home else matches['AF']
    tempo_index = (shots + corners + fouls).median()  # ✅ robustní

    # Agresivita (relativní podle celé ligy)
    yellow_cards = matches['HY'] if is_home else matches['AY']
    red_cards = matches['HR'] if is_home else matches['AR']
    aggressiveness_index = ((yellow_cards + 2 * red_cards + fouls).sum()) / len(matches)

    # 🧠 Percentilové škálování agresivity
    league_aggs = []
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for t in all_teams:
        home = df[df['HomeTeam'] == t]
        away = df[df['AwayTeam'] == t]
        if not home.empty or not away.empty:
            f = pd.concat([home['HF'], away['AF']])
            y = pd.concat([home['HY'], away['AY']])
            r = pd.concat([home['HR'], away['AR']])
            agg = ((y + 2 * r + f).sum()) / (len(home) + len(away))
            league_aggs.append(agg)

    p20 = np.percentile(league_aggs, 20)
    p40 = np.percentile(league_aggs, 40)
    p60 = np.percentile(league_aggs, 60)
    p80 = np.percentile(league_aggs, 80)

    if aggressiveness_index < p20:
        aggressiveness_rating = "🕊️ velmi klidné"
    elif aggressiveness_index < p40:
        aggressiveness_rating = "🟢 korektní"
    elif aggressiveness_index < p60:
        aggressiveness_rating = "🟡 tvrdší styl"
    elif aggressiveness_index < p80:
        aggressiveness_rating = "🔴 tvrdý styl"
    else:
        aggressiveness_rating = "🟥 velmi tvrdý"

    # Tempo soupeřů
    opponent_shots = matches['AS'] if is_home else matches['HS']
    opponent_corners = matches['AC'] if is_home else matches['HC']
    opponent_fouls = matches['AF'] if is_home else matches['HF']
    similar_opponents_tempo = (opponent_shots + opponent_corners + opponent_fouls).median()

    # Percentil tempa v lize
    all_tempos = []
    for t in all_teams:
        home = df[df['HomeTeam'] == t]
        away = df[df['AwayTeam'] == t]
        if not home.empty or not away.empty:
            s = pd.concat([home['HS'], away['AS']])
            c = pd.concat([home['HC'], away['AC']])
            f = pd.concat([home['HF'], away['AF']])
            all_tempos.append((s + c + f).median())

    percentile = round(sum(t < tempo_index for t in all_tempos) / len(all_tempos) * 100, 1)

    if percentile >= 80:
        rating = "⚡ velmi rychlé"
    elif percentile >= 40:
        rating = "🎯 střední tempo"
    elif percentile >= 10:
        rating = "💤 pomalé"
    else:
        rating = "🪨 velmi pomalé"

    # ⚖️ IMBALANCE – jen proti soupeřům stejné síly
    

    opponent_class = classify_team_strength(df, team=matches[opp_col].iloc[0])
    filtered = matches.copy()
    filtered["Opponent"] = matches[opp_col]
    filtered["OpponentClass"] = filtered["Opponent"].apply(lambda t: classify_team_strength(df, t))
    filtered = filtered[filtered["OpponentClass"] == opponent_class]
    if filtered.empty:
        filtered = matches  # fallback

    goals_for = filtered['FTHG'] if is_home else filtered['FTAG']
    goals_against = filtered['FTAG'] if is_home else filtered['FTHG']
    shots_for = filtered['HS'] if is_home else filtered['AS']
    shots_against = filtered['AS'] if is_home else filtered['HS']

    goal_diff = (goals_for - goals_against).median()
    shot_diff = (shots_for - shots_against).median()

    min_threshold = 0.3
    if goal_diff > min_threshold and shot_diff > min_threshold:
        imbalance_type = "📈 Dominantní"
    elif goal_diff < -min_threshold and shot_diff < -min_threshold:
        imbalance_type = "📉 Trpící"
    else:
        imbalance_type = "⚖️ Neurčitá"

    imbalance = abs(goal_diff) + abs(shot_diff)

    return {
        "tempo": round(tempo_index, 1),
        "percentile": percentile,
        "rating": rating,
        "imbalance": round(imbalance, 2),
        "imbalance_type": imbalance_type,
        "aggressiveness_index": round(aggressiveness_index, 2),
        "aggressiveness_rating": aggressiveness_rating,
        "similar_opponents_tempo": round(similar_opponents_tempo, 1)
    }



def style_team_table(df):
    def style_status(val):
        emoji = "🟢" if val == "Forma" else "🟡" if val == "Průměr" else "🔴"
        return f"{emoji} {val}"

    def color_performance(val):
        if "Nadprůměr" in val:
            return "color: green"
        elif "Nízký" in val or "Slabý" in val:
            return "color: red"
        return "color: black"

    def color_momentum(val):
        if "Pozitivní" in val:
            return "background-color: #d1fae5"
        elif "Negativní" in val:
            return "background-color: #fee2e2"
        return ""

    styled_df = df.copy()
    styled_df["Status"] = styled_df["Status"].apply(style_status)

    return styled_df.style.applymap(color_performance, subset=["Overperformance"])\
                          .applymap(color_momentum, subset=["Momentum"])




def calculate_match_style_score_per_match(df: pd.DataFrame) -> pd.DataFrame:
    """Přidá komplexní metriky stylu hry do každého zápasu."""
    df = prepare_df(df)

    # Základní složky
    df['Tempo'] = df['HS'] + df['AS'] + df['HC'] + df['AC'] + df['HF'] + df['AF']
    df['Goly'] = df['FTHG'] + df['FTAG']
    df['Konverze'] = (df['FTHG'] + df['FTAG']) / (df['HST'] + df['AST']).replace(0, 0.1)
    df['Agrese'] = df['HY'] + df['AY'] + 2 * (df['HR'] + df['AR']) + df['HF'] + df['AF']

    # Normalizace pro složky do 0–1
    for col in ['Tempo', 'Goly', 'Konverze', 'Agrese']:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col + "_norm"] = (df[col] - min_val) / (max_val - min_val + 1e-5)

    # Výpočet finálního skóre (0–100)
    df['MatchStyleScore'] = (
        0.55 * df['Tempo_norm'] +
        0.25 * df['Goly_norm'] +
        0.10 * df['Konverze_norm'] +
        0.10 * df['Agrese_norm']
    ) * 100

    return df


def calculate_gii_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vrací GII index (Game Intensity Index) pro každý zápas
    na základě MatchStyleScore.
    """
    df = calculate_match_style_score_per_match(df)

    gii_mean = df['MatchStyleScore'].mean()
    gii_std = df['MatchStyleScore'].std()

    df['GII'] = (df['MatchStyleScore'] - gii_mean) / (gii_std + 1e-5)

    return df


def calculate_gii_for_team(df: pd.DataFrame, team: str) -> float:
    """Spočítá GII pro tým = průměr (střely + rohy + fauly) na zápas."""
    df = prepare_df(df)

    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]

    if home.empty and away.empty:
        return 0.0

    shots = pd.concat([home['HS'], away['AS']])
    corners = pd.concat([home['HC'], away['AC']])
    fouls = pd.concat([home['HF'], away['AF']])

    gii = (shots + corners + fouls).mean()

    return round(gii, 2)

def calculate_league_gii(df: pd.DataFrame) -> float:
    """Spočítá průměrné GII v celé lize."""
    df = prepare_df(df)

    shots = df['HS'] + df['AS']
    corners = df['HC'] + df['AC']
    fouls = df['HF'] + df['AF']

    total_actions = shots + corners + fouls
    matches = len(df)

    league_gii = total_actions.sum() / matches

    return round(league_gii, 2)

def calculate_attack_volume(df: pd.DataFrame, team: str) -> float:
    """Spočítá objem útoků = (střely + rohy) na zápas."""
    df = prepare_df(df)

    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]

    if home.empty and away.empty:
        return 0.0

    shots = pd.concat([home['HS'], away['AS']])
    corners = pd.concat([home['HC'], away['AC']])

    volume = (shots + corners).mean()

    return round(volume, 2)

def calculate_attack_efficiency(df: pd.DataFrame, team: str) -> float:
    """Spočítá efektivitu útoků = střely na bránu / střely."""
    df = prepare_df(df)

    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]

    if home.empty and away.empty:
        return 0.0

    shots = pd.concat([home['HS'], away['AS']])
    shots_on_target = pd.concat([home['HST'], away['AST']])

    if shots.sum() == 0:
        return 0.0

    efficiency = shots_on_target.sum() / shots.sum()

    return round(efficiency, 2)

def calculate_full_attacking_pressure(df: pd.DataFrame, team: str) -> float:
    """Spočítá kombinovaný útočný tlak: Attack Volume × Attack Efficiency."""
    volume = calculate_attack_volume(df, team)
    efficiency = calculate_attack_efficiency(df, team)

    attacking_pressure = volume * efficiency

    return round(attacking_pressure, 2)

def get_team_style_vs_opponent_type(df: pd.DataFrame, team: str, opponent_team: str) -> float:
    """
    Vrací průměrný MatchStyleScore (nebo GII), když tým hrál proti soupeřům stejné kategorie jako `opponent_team`.
    """
    

    df = calculate_match_style_score_per_match(df)
    df = prepare_df(df)

    # Zjisti sílu soupeře (silný, průměrný, slabý)
    opponent_class = classify_team_strength(df, opponent_team)

    # Najdi všechny zápasy daného týmu
    matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
    matches["Opponent"] = matches.apply(
        lambda row: row["AwayTeam"] if row["HomeTeam"] == team else row["HomeTeam"], axis=1
    )

    # Zjisti sílu každého soupeře
    matches["OpponentClass"] = matches["Opponent"].apply(lambda opp: classify_team_strength(df, opp))

    # Filtrovat pouze zápasy proti stejné kategorii
    filtered = matches[matches["OpponentClass"] == opponent_class]

    if filtered.empty:
        print(f"[{team}] ⚠️ Žádné zápasy proti {opponent_class} týmům.")
        return None

    # Výpočet průměrného stylu (MatchStyleScore)
    return round(filtered["MatchStyleScore"].mean(), 1)


def get_team_average_gii(df: pd.DataFrame) -> dict:
    """
    Vrací dict: tým -> průměrné Z-skóre GII (intenzity) na základě zápasů.
    Používá předzpracovaný 'MatchStyleScore' a Z-skórovou normalizaci.
    """
    df = calculate_match_style_score_per_match(df)

    # Výpočet Z-skóre GII z MatchStyleScore
    mean = df['MatchStyleScore'].mean()
    std = df['MatchStyleScore'].std()
    df['GII'] = (df['MatchStyleScore'] - mean) / (std + 1e-5)

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    gii_dict = {}

    for team in teams:
        gii_home = df[df['HomeTeam'] == team]['GII']
        gii_away = df[df['AwayTeam'] == team]['GII']
        values = pd.concat([gii_home, gii_away])
        if len(values) >= 3:  # min počet zápasů
            gii_dict[team] = round(values.mean(), 2)
        else:
            gii_dict[team] = None  # nebo 0.0, nebo nezobrazit

    return gii_dict


def calculate_team_styles(df: pd.DataFrame) -> tuple:
    """Vrací DataFramy ofenzivního a defenzivního stylu týmů."""
    df = prepare_df(df)
    offensive_style = []
    defensive_style = []
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    for team in teams:
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]
        shots = pd.concat([home['HS'], away['AS']]).mean()
        sot = pd.concat([home['HST'], away['AST']]).mean()
        corners = pd.concat([home['HC'], away['AC']]).mean()
        xg = (sot * (pd.concat([home['FTHG'], away['FTAG']]).sum() / sot)) if sot > 0 else 0
        fouls = pd.concat([home['HF'], away['AF']]).mean()

        offensive_index = shots * 0.25 + sot * 0.25 + corners * 0.2 + xg * 0.2 + fouls * 0.1
        defensive_index = (1 / (shots + 1)) * 0.3 + (1 / (sot + 1)) * 0.25 + (1 / (corners + 1)) * 0.2 + (1 / (xg + 0.1)) * 0.15 + (1 / (fouls + 1)) * 0.1

        offensive_style.append({"Tým": team, "Ofenzivní styl index": round(offensive_index, 2)})
        defensive_style.append({"Tým": team, "Defenzivní styl index": round(defensive_index, 2)})

    off_df = pd.DataFrame(offensive_style).sort_values("Ofenzivní styl index", ascending=False).reset_index(drop=True)
    def_df = pd.DataFrame(defensive_style).sort_values("Defenzivní styl index", ascending=False).reset_index(drop=True)

    return off_df, def_df

def intensity_score_to_emoji(score: float) -> str:
    """Převede skóre GII na emoji."""
    if score > 1.0:
        return "🔥"
    elif score > 0.3:
        return "⚡"
    elif score > -0.3:
        return "➖"
    else:
        return "❄️"

def form_points_to_emoji(avg_points: float) -> str:
    """Vrací emoji podle průměrného počtu bodů za zápas."""
    if avg_points > 2.5:
        return "🔥🔥🔥"
    elif avg_points > 2.0:
        return "🔥🔥"
    elif avg_points > 1.5:
        return "🔥"
    elif avg_points > 1.2:
        return "💤"
    elif avg_points > 0.7:
        return "❄️"
    elif avg_points > 0.5:
        return "❄️❄️"
    else:
        return "❄️❄️❄️"
    
def tempo_tag(tempo_value: float) -> str:
    """
    Vygeneruje barevný HTML badge podle hodnoty tempa.
    """
    if tempo_value >= 65:
        color = "#FF4B4B"  # červená – extrémní tempo
        label = "🚀 Extrémní tempo"
    elif tempo_value >= 50:
        color = "#FF8800"  # oranžová – vysoké tempo
        label = "🔥 Vysoké tempo"
    elif tempo_value >= 35:
        color = "#FACC15"  # žlutá – vyrovnané tempo
        label = "⚖️ Vyrovnané tempo"
    elif tempo_value >= 20:
        color = "#3B82F6"  # modrá – pomalé tempo
        label = "😴 Pomalé tempo"
    else:
        color = "#6B7280"  # šedá – totální nuda
        label = "💤 Totální nuda"

    return f"""
    <div style='
        display: inline-block;
        background-color: {color};
        color: white;
        padding: 6px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 10px;
    '>{label} ({tempo_value})</div>
    """


def expected_match_style_score(df: pd.DataFrame, home_team: str, away_team: str, elo_dict: dict) -> float:
    """Spočítá očekávaný styl zápasu (průměr temp domácích a hostů)."""
    home_tempo = calculate_match_tempo(df, home_team, elo_dict.get(away_team, 1500), is_home=True, elo_dict=elo_dict)
    away_tempo = calculate_match_tempo(df, away_team, elo_dict.get(home_team, 1500), is_home=False, elo_dict=elo_dict)

    home_tempo_value = home_tempo["tempo"]
    away_tempo_value = away_tempo["tempo"]


    return round((home_tempo_value  + away_tempo_value) / 2, 1)

def expected_match_tempo(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    elo_dict: dict,
    home_exp: float,
    away_exp: float,
    xg_home: float,
    xg_away: float,
    last_n: int = 10
) -> float:
    """
    Vylepšené očekávané tempo zápasu, bere v úvahu i sílu útočných fází.
    """
    df = prepare_df(df)

    elo_away = elo_dict.get(away_team, 1500)
    elo_home = elo_dict.get(home_team, 1500)

    home_tempo_dict = calculate_match_tempo(df, home_team, opponent_elo=elo_away, is_home=True, elo_dict=elo_dict, last_n=last_n)
    away_tempo_dict = calculate_match_tempo(df, away_team, opponent_elo=elo_home, is_home=False, elo_dict=elo_dict, last_n=last_n)

    base_tempo = (home_tempo_dict["tempo"] + away_tempo_dict["tempo"]) / 2

    # Gólové ukazatele
    goal_potential = home_exp + away_exp
    xg_potential = xg_home + xg_away

    # Bonus za extrémní gólový potenciál
    high_scoring_bonus = 5 if goal_potential >= 3.0 else 0

    # Kombinace složek (násobení pro větší rozptyl)
    combined_score = (
        base_tempo * 0.3 +
        goal_potential * 20 * 0.35 +
        xg_potential * 20 * 0.35 +
        high_scoring_bonus
    )

    return round(combined_score, 1)



def team_vs_similar_opponents_tempo(df: pd.DataFrame, team: str, opponent_elo: float, is_home: bool, elo_dict: dict, last_n: int = 10) -> float:
    """Spočítá průměrné tempo týmu proti soupeřům s podobným ELO."""
    df = prepare_df(df)

    tempo_dict = calculate_match_tempo(df, team, opponent_elo=opponent_elo, is_home=is_home, elo_dict=elo_dict, last_n=last_n)

    return tempo_dict["tempo"]

def tempo_to_emoji(tempo_value: float) -> str:
    """
    Vrací emoji podle tempa zápasu.
    """
    if tempo_value >= 65:
        return "🚀 Extrémní tempo"
    elif tempo_value >= 50:
        return "🔥 Vysoké tempo"
    elif tempo_value >= 35:
        return "⚖️ Vyrovnané tempo"
    elif tempo_value >= 20:
        return "😴 Pomalý zápas"
    else:
        return "💤 Totální nuda"


