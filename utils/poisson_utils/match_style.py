import pandas as pd
import numpy as np

from .core import prepare_df

def calculate_match_tempo(df: pd.DataFrame, team: str, opponent_elo: float, is_home: bool, elo_dict: dict, last_n: int = 10) -> dict:
    """Spočítá tempo zápasu, agresivitu a dominanci týmu na základě posledních zápasů proti podobně silným soupeřům."""
    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    opp_col = 'AwayTeam' if is_home else 'HomeTeam'
    df = prepare_df(df)

    df['OppELO'] = df[opp_col].map(elo_dict)
    df['EloDiff'] = abs(df['OppELO'] - opponent_elo)
    matches = df[df[team_col] == team].sort_values('EloDiff').head(last_n)

    if matches.empty:
        matches = df[df[team_col] == team].sort_values("Date", ascending=False).head(last_n)

    if matches.empty:
        return {
            "tempo": 0,
            "percentile": 0,
            "rating": "N/A",
            "imbalance": 0.0,
            "imbalance_type": "N/A",
            "aggressiveness_index": 0.0,
            "aggressiveness_rating": "N/A",
            "similar_opponents_tempo": 0.0   # <-- nový klíč
        }

    shots = matches['HS'] if is_home else matches['AS']
    corners = matches['HC'] if is_home else matches['AC']
    fouls = matches['HF'] if is_home else matches['AF']
    tempo_index = (shots + corners + fouls).mean()

    yellow_cards = matches['HY'] if is_home else matches['AY']
    red_cards = matches['HR'] if is_home else matches['AR']
    aggressiveness_index = ((yellow_cards + 2 * red_cards + fouls).sum()) / len(matches)

    ### 📌 Tady vypočítáme průměrné tempo proti podobným soupeřům
    opponent_shots = matches['AS'] if is_home else matches['HS']
    opponent_corners = matches['AC'] if is_home else matches['HC']
    opponent_fouls = matches['AF'] if is_home else matches['HF']
    similar_opponents_tempo = (opponent_shots + opponent_corners + opponent_fouls).mean()

    ### 📌 Percentil tempa v celé lize
    all_tempos = []
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for t in all_teams:
        home = df[df['HomeTeam'] == t]
        away = df[df['AwayTeam'] == t]
        if not home.empty and not away.empty:
            s = pd.concat([home['HS'], away['AS']])
            c = pd.concat([home['HC'], away['AC']])
            f = pd.concat([home['HF'], away['AF']])
            all_tempos.append((s + c + f).mean())

    percentile = round(sum(t < tempo_index for t in all_tempos) / len(all_tempos) * 100, 1)

    if percentile >= 80:
        rating = "⚡ velmi rychlé"
    elif percentile >= 40:
        rating = "🎯 střední tempo"
    elif percentile >= 10:
        rating = "💤 pomalé"
    else:
        rating = "🪨 velmi pomalé"

    if is_home:
        goals_for = matches['FTHG']
        goals_against = matches['FTAG']
    else:
        goals_for = matches['FTAG']
        goals_against = matches['FTHG']

    shots_for = matches['HS'] if is_home else matches['AS']
    shots_against = matches['AS'] if is_home else matches['HS']

    goal_diff = (goals_for - goals_against).mean()
    shot_diff = (shots_for - shots_against).mean()
    imbalance = (abs(goal_diff) + abs(shot_diff))

    if goal_diff > 0 and shot_diff > 0:
        imbalance_type = "📈 Dominantní"
    elif goal_diff < 0 and shot_diff < 0:
        imbalance_type = "📉 Trpící"
    else:
        imbalance_type = "⚖️ Neurčitá"

    if aggressiveness_index < 10:
        aggressiveness_rating = "🕊️ velmi klidné"
    elif aggressiveness_index < 13:
        aggressiveness_rating = "🟢 korektní"
    elif aggressiveness_index < 18:
        aggressiveness_rating = "🟡 tvrdší zápas"
    elif aggressiveness_index < 22:
        aggressiveness_rating = "🔴 velmi tvrdý"
    else:
        aggressiveness_rating = "🟥 extrémní zákroky"

    return {
        "tempo": round(tempo_index, 1),
        "percentile": percentile,
        "rating": rating,
        "imbalance": round(imbalance, 2),
        "imbalance_type": imbalance_type,
        "aggressiveness_index": round(aggressiveness_index, 2),
        "aggressiveness_rating": aggressiveness_rating,
        "similar_opponents_tempo": round(similar_opponents_tempo, 1)   # <-- tady vracíme nový výstup
    }




def calculate_match_style_score_per_match(df: pd.DataFrame) -> pd.DataFrame:
    """Přidá metriky o tempu zápasu a stylu hry do datasetu."""
    df = prepare_df(df)

    df['match_tempo'] = df['HS'] + df['AS'] + df['HF'] + df['AF']
    df['attacking_pressure'] = (df['HST'] + df['AST']) / (df['HS'] + df['AS'] + 1)
    df['disciplinary_index'] = (df['HY'] + df['AY'] + 2 * (df['HR'] + df['AR'])) / (df['HF'] + df['AF'] + 1)

    return df

def calculate_gii_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Vrací GII index (Game Intensity Index) pro každý zápas."""
    df = calculate_match_style_score_per_match(df)

    gii_mean = df['match_tempo'].mean()
    gii_std = df['match_tempo'].std()

    df['GII'] = (df['match_tempo'] - gii_mean) / gii_std

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



def get_team_average_gii(df: pd.DataFrame) -> dict:
    """Vrací dictionary tým -> průměr GII ze zápasů."""
    df = calculate_gii_zscore(df)
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    gii_dict = {}

    for team in teams:
        home_gii = df[df['HomeTeam'] == team]['GII']
        away_gii = df[df['AwayTeam'] == team]['GII']
        avg_gii = pd.concat([home_gii, away_gii]).mean()
        gii_dict[team] = round(avg_gii, 2)

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

def expected_match_style_score(df: pd.DataFrame, home_team: str, away_team: str, elo_dict: dict) -> float:
    """Spočítá očekávaný styl zápasu (průměr temp domácích a hostů)."""
    home_tempo = calculate_match_tempo(df, home_team, elo_dict.get(away_team, 1500), is_home=True, elo_dict=elo_dict)
    away_tempo = calculate_match_tempo(df, away_team, elo_dict.get(home_team, 1500), is_home=False, elo_dict=elo_dict)

    home_tempo_value = home_tempo["tempo"]
    away_tempo_value = away_tempo["tempo"]


    return round((home_tempo_value  + away_tempo_value) / 2, 1)

def expected_match_tempo(df: pd.DataFrame, home_team: str, away_team: str, elo_dict: dict, last_n: int = 10) -> float:
    """Spočítá očekávané tempo zápasu mezi dvěma týmy na základě posledních zápasů proti podobným soupeřům."""
    df = prepare_df(df)

    elo_away = elo_dict.get(away_team, 1500)
    elo_home = elo_dict.get(home_team, 1500)

    # Tempo domácího týmu proti soupeřům podobným hostujícímu týmu
    home_tempo_dict = calculate_match_tempo(df, home_team, opponent_elo=elo_away, is_home=True, elo_dict=elo_dict, last_n=last_n)

    # Tempo hostujícího týmu proti soupeřům podobným domácímu týmu
    away_tempo_dict = calculate_match_tempo(df, away_team, opponent_elo=elo_home, is_home=False, elo_dict=elo_dict, last_n=last_n)

    expected_tempo = (home_tempo_dict["tempo"] + away_tempo_dict["tempo"]) / 2

    return round(expected_tempo, 1)

def team_vs_similar_opponents_tempo(df: pd.DataFrame, team: str, opponent_elo: float, is_home: bool, elo_dict: dict, last_n: int = 10) -> float:
    """Spočítá průměrné tempo týmu proti soupeřům s podobným ELO."""
    df = prepare_df(df)

    tempo_dict = calculate_match_tempo(df, team, opponent_elo=opponent_elo, is_home=is_home, elo_dict=elo_dict, last_n=last_n)

    return tempo_dict["tempo"]

def tempo_to_emoji(tempo_value: float) -> str:
    """Přiřadí emoji a slovní hodnocení k dané hodnotě tempa."""
    if tempo_value >= 50:
        return "⚡ velmi rychlé"
    elif tempo_value >= 35:
        return "🎯 střední tempo"
    elif tempo_value >= 20:
        return "💤 pomalé"
    else:
        return "🪨 velmi pomalé"

