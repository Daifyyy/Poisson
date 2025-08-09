import pandas as pd
import numpy as np
from scipy.stats import poisson
import streamlit as st
from .data import prepare_df, get_last_n_matches
from .stats import calculate_points
from .prediction import poisson_over25_probability, expected_goals_vs_similar_elo_weighted 
from .xg import calculate_team_pseudo_xg
from utils.utils_warnings import detect_overperformance_and_momentum

def calculate_form_emojis(df: pd.DataFrame, days: int = 31) -> dict:
    """Vrací dictionary: tým -> emoji reprezentace formy."""
    from .match_style import form_points_to_emoji
    form_dict = calculate_recent_form(df, days=days)
    form_emojis = {}
    for team, avg_points in form_dict.items():
        form_emojis[team] = form_points_to_emoji(avg_points)
    return form_emojis

def calculate_conceded_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Vrací DataFrame s průměrným počtem obdržených gólů pro každý tým."""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    conceded_stats = []
    for team in teams:
        home = df[df['AwayTeam'] == team]
        away = df[df['HomeTeam'] == team]
        goals_against = pd.concat([home['FTHG'], away['FTAG']]).mean()
        conceded_stats.append({"Tým": team, "Obdržené góly": round(goals_against, 2)})
    return pd.DataFrame(conceded_stats).sort_values("Obdržené góly", ascending=False).reset_index(drop=True)

def calculate_recent_team_form(df: pd.DataFrame, last_n: int = 5) -> pd.DataFrame:
    """Vrací DataFrame s průměrem bodů a formou (emoji) za posledních N zápasů pro každý tým."""
    from .match_style import form_points_to_emoji
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    form_stats = []
    for team in teams:
        recent_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values("Date").tail(last_n)
        total_points = 0
        for _, row in recent_matches.iterrows():
            is_home = row['HomeTeam'] == team
            points = calculate_points(row, is_home)
            total_points += points
        avg_points = total_points / last_n
        form_stats.append({"Tým": team, "Body/zápas": avg_points})
    form_df = pd.DataFrame(form_stats)
    form_df["Form"] = form_df["Body/zápas"].apply(form_points_to_emoji)
    return form_df.sort_values("Body/zápas").reset_index(drop=True)

import numpy as np
from scipy.stats import poisson

def calculate_expected_and_actual_points(df: pd.DataFrame) -> dict:
    """Spočítá skutečné a očekávané body týmů na základě proxy xG modelu (poměr střel na bránu ke střelám)."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    results = {}

    for team in teams:
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]
        all_matches = pd.concat([home, away])

        if all_matches.empty:
            results[team] = {
                "points": 0,
                "points_per_game": 0,
                "matches": 0,
                "expected_points": 0
            }
            continue

        # Skutečné body
        home_points = sum(3 if row['FTHG'] > row['FTAG'] else 1 if row['FTHG'] == row['FTAG'] else 0 for _, row in home.iterrows())
        away_points = sum(3 if row['FTAG'] > row['FTHG'] else 1 if row['FTAG'] == row['FTHG'] else 0 for _, row in away.iterrows())
        total_points = home_points + away_points
        num_matches = len(home) + len(away)

        # Expected points (xP) Calculation
        xP = 0
        for _, row in all_matches.iterrows():
            if row['HomeTeam'] == team:
                xg_for = (row['HST'] / row['HS']) if row['HS'] > 0 else 0.1
                xg_against = (row['AST'] / row['AS']) if row['AS'] > 0 else 0.1
                team_is_home = True
            elif row['AwayTeam'] == team:
                xg_for = (row['AST'] / row['AS']) if row['AS'] > 0 else 0.1
                xg_against = (row['HST'] / row['HS']) if row['HS'] > 0 else 0.1
                team_is_home = False
            else:
                continue

            max_goals = 6
            probs = [[poisson.pmf(i, xg_for) * poisson.pmf(j, xg_against) for j in range(max_goals)] for i in range(max_goals)]

            for i in range(max_goals):
                for j in range(max_goals):
                    p = probs[i][j]
                    if team_is_home:
                        if i > j:
                            xP += 3 * p
                        elif i == j:
                            xP += 1 * p
                    else:
                        if j > i:
                            xP += 3 * p
                        elif i == j:
                            xP += 1 * p

        results[team] = {
            "points": total_points,
            "points_per_game": round(total_points / num_matches, 2) if num_matches > 0 else 0,
            "matches": num_matches,
            "expected_points": round(xP, 2)
        }

    return results


def analyze_opponent_strength(df: pd.DataFrame, team: str, is_home: bool = True) -> dict:
    """Analyzuje sílu soupeřů podle výsledků."""
    df = prepare_df(df)

    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    opp_col = 'AwayTeam' if is_home else 'HomeTeam'
    goals_col = 'FTHG' if is_home else 'FTAG'
    shots_col = 'HS' if is_home else 'AS'

    team_matches = df[df[team_col] == team]

    avg_goals_per_team = {}
    for t in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        home_goals = df[df['HomeTeam'] == t]['FTHG'].mean()
        away_goals = df[df['AwayTeam'] == t]['FTAG'].mean()
        avg_goals_per_team[t] = np.nanmean([home_goals, away_goals])

    sorted_teams = sorted(avg_goals_per_team.items(), key=lambda x: x[1], reverse=True)
    total = len(sorted_teams)
    top_teams = set(t for t, _ in sorted_teams[:int(total * 0.3)])
    bottom_teams = set(t for t, _ in sorted_teams[-int(total * 0.3):])
    middle_teams = set(avg_goals_per_team.keys()) - top_teams - bottom_teams

    performance = {"strong": [], "average": [], "weak": []}

    for _, row in team_matches.iterrows():
        opponent = row[opp_col]
        goals = row[goals_col]
        shots = row[shots_col]
        points = calculate_points(row, is_home)

        data_point = {'goals': goals, 'shots': shots, 'points': points}

        if opponent in top_teams:
            performance['strong'].append(data_point)
        elif opponent in bottom_teams:
            performance['weak'].append(data_point)
        else:
            performance['average'].append(data_point)

    def summarize(data):
        if not data:
            return {'matches': 0, 'goals': 0, 'con_rate': 0, 'xP': 0}
        matches = len(data)
        goals = np.mean([d['goals'] for d in data])
        shots = np.mean([d['shots'] for d in data])
        con_rate = round(goals / shots, 2) if shots > 0 else 0
        xP = round(np.mean([d['points'] for d in data]), 2)
        return {'matches': matches, 'goals': round(goals, 2), 'con_rate': con_rate, 'xP': xP}

    return {
        'vs_strong': summarize(performance['strong']),
        'vs_average': summarize(performance['average']),
        'vs_weak': summarize(performance['weak']),
    }

def get_head_to_head_stats(df: pd.DataFrame, home_team: str, away_team: str, last_n: int = 5) -> dict:
    """Vrací head-to-head statistiky za posledních N zápasů mezi dvěma týmy."""
    df = prepare_df(df)

    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].sort_values('Date', ascending=False).head(last_n)

    if h2h.empty:
        return None

    results = {
        "matches": len(h2h),
        "home_wins": 0,
        "away_wins": 0,
        "draws": 0,
        "avg_goals": round((h2h['FTHG'] + h2h['FTAG']).mean(), 2),
        "btts_pct": round(100 * h2h.apply(lambda r: r['FTHG'] > 0 and r['FTAG'] > 0, axis=1).mean(), 1),
        "over25_pct": round(100 * ((h2h['FTHG'] + h2h['FTAG']) > 2.5).mean(), 1)
    }

    for _, row in h2h.iterrows():
        if row['FTHG'] == row['FTAG']:
            results['draws'] += 1
        elif (row['HomeTeam'] == home_team and row['FTHG'] > row['FTAG']) or \
             (row['AwayTeam'] == home_team and row['FTAG'] > row['FTHG']):
            results['home_wins'] += 1
        else:
            results['away_wins'] += 1

    return results

    
def merged_home_away_opponent_form(df: pd.DataFrame, team: str) -> dict:
    """Vrací kombinovanou domácí a venkovní formu týmu vůči silným, průměrným a slabým soupeřům."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    team_avg_goals = {}
    for t in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        home_g = df[df['HomeTeam'] == t]['FTHG'].mean()
        away_g = df[df['AwayTeam'] == t]['FTAG'].mean()
        team_avg_goals[t] = np.nanmean([home_g, away_g])

    sorted_teams = sorted(team_avg_goals.items(), key=lambda x: x[1], reverse=True)
    top = set(t for t, _ in sorted_teams[:int(len(sorted_teams) * 0.3)])
    bottom = set(t for t, _ in sorted_teams[-int(len(sorted_teams) * 0.3):])
    middle = set(team_avg_goals.keys()) - top - bottom

    def summarize(matches, is_home):
        if matches.empty:
            return {"Z": 0, "G": 0, "OG": 0, "S": 0, "SOT": 0, "xG": 0, "PTS": 0, "CS": 0}
        goals_for = matches["FTHG"] if is_home else matches["FTAG"]
        goals_against = matches["FTAG"] if is_home else matches["FTHG"]
        shots = matches["HS"] if is_home else matches["AS"]
        sot = matches["HST"] if is_home else matches["AST"]
        conv = goals_for.mean() / sot.mean() if sot.mean() > 0 else 0
        xg = round(sot.mean() * conv, 2)
        points = matches.apply(
            lambda r: 3 if (r["FTHG"] > r["FTAG"] if is_home else r["FTAG"] > r["FTHG"]) else 1 if r["FTHG"] == r["FTAG"] else 0,
            axis=1
        ).mean()
        clean_sheets = (goals_against == 0).sum()
        cs_percent = round(100 * clean_sheets / len(matches), 1)
        return {
            "Zápasy": len(matches),
            "Góly": round(goals_for.mean(), 2),
            "Obdržené": round(goals_against.mean(), 2),
            "Střely": round(shots.mean(), 1),
            "Na branku": round(sot.mean(), 1),
            "xG": xg,
            "Body/zápas": round(points, 2),
            "Čistá konta %": cs_percent
        }

    result = {}

    for label, group in [("💪 Silní", top), ("⚖️ Průměrní", middle), ("🪶 Slabí", bottom)]:
        home_matches = df[(df['HomeTeam'] == team) & (df['AwayTeam'].isin(group))]
        away_matches = df[(df['AwayTeam'] == team) & (df['HomeTeam'].isin(group))]

        home_stats = summarize(home_matches, is_home=True)
        away_stats = summarize(away_matches, is_home=False)

        result[label] = {
            "Zápasy": f"{home_stats['Zápasy']} / {away_stats['Zápasy']}",
            "Góly": f"{home_stats['Góly']} / {away_stats['Góly']}",
            "Obdržené": f"{home_stats['Obdržené']} / {away_stats['Obdržené']}",
            "Střely": f"{home_stats['Střely']} / {away_stats['Střely']}",
            "Na branku": f"{home_stats['Na branku']} / {away_stats['Na branku']}",
            "xG": f"{home_stats['xG']} / {away_stats['xG']}",
            "Body/zápas": f"{home_stats['Body/zápas']} / {away_stats['Body/zápas']}",
            "Čistá konta %": f"{home_stats['Čistá konta %']} / {away_stats['Čistá konta %']}"
        }

    return result

def calculate_recent_form(df: pd.DataFrame, days: int = 31) -> dict:
    """Vrací dictionary: tým -> průměr bodů za posledních N dní."""
    from .data import prepare_df
    from .stats import calculate_points

    df = prepare_df(df)
    latest_date = df['Date'].max()
    recent_df = df[df['Date'] >= latest_date - pd.Timedelta(days=days)]

    teams = pd.concat([recent_df['HomeTeam'], recent_df['AwayTeam']]).unique()
    form = {}

    for team in teams:
        matches = recent_df[(recent_df['HomeTeam'] == team) | (recent_df['AwayTeam'] == team)]

        if matches.empty:
            form[team] = 0
            continue

        points = []
        for _, row in matches.iterrows():
            is_home = row['HomeTeam'] == team
            points.append(calculate_points(row, is_home))

        avg_points = np.mean(points) if points else 0
        form[team] = round(avg_points, 2)

    return form

def get_team_card_stats(df, team):
    """
    Spočítá celkový počet žlutých a červených karet daného týmu v zadaném dataframe.
    
    Vrací slovník:
    {
        "yellow": int,
        "red": int,
        "fouls": int
    }
    """
    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]

    total_yellow = home['HY'].sum() + away['AY'].sum()
    total_red = home['HR'].sum() + away['AR'].sum()
    total_fouls = home['HF'].sum() + away['AF'].sum()

    return {
        "yellow": total_yellow,
        "red": total_red,
        "fouls": total_fouls
    }


def calculate_team_home_advantage(df, team: str) -> float:
    """
    Spočítá relativní domácí výhodu daného týmu vůči ligovému průměru.
    Výstupem je upravený home advantage v rozsahu ±0.3 (většinou).
    """
    home_avg = df[df['HomeTeam'] == team]['FTHG'].mean()
    away_avg = df[df['AwayTeam'] == team]['FTAG'].mean()
    team_diff = home_avg - away_avg

    league_home_avg = df['FTHG'].mean()
    league_away_avg = df['FTAG'].mean()
    league_diff = league_home_avg - league_away_avg

    if pd.isna(team_diff) or pd.isna(league_diff) or league_diff == 0:
        return 0.0

    home_adv_ratio = team_diff / league_diff
    home_adv_scaled = league_diff * home_adv_ratio * 0.5  # tlumený koeficient

    return round(home_adv_scaled, 2)

def expected_goals_weighted_by_home_away(df, home_team, away_team, elo_dict) -> tuple:
    """
    Rozšířená verze výpočtu očekávaných gólů, která respektuje domácí vs venkovní výkonnost
    a dynamicky dopočítává faktor domácí výhody pomocí funkce calculate_team_home_advantage().
    """
    
    df = prepare_df(df)

    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)

    df_hist = df[df['Date'] < one_year_ago]
    df_season = df[df['Date'] >= one_year_ago]
    df_last10 = df[
        (df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team) |
        (df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)
    ].sort_values('Date').tail(10)

    def filter_by_elo(sub_df, team, is_home, opponent_elo, n=10):
        team_col = 'HomeTeam' if is_home else 'AwayTeam'
        opp_col = 'AwayTeam' if is_home else 'HomeTeam'
        gf_col = 'FTHG' if is_home else 'FTAG'
        ga_col = 'FTAG' if is_home else 'FTHG'

        matches = sub_df[sub_df[team_col] == team].copy()
        matches['OppELO'] = matches[opp_col].map(elo_dict)
        matches['EloDiff'] = abs(matches['OppELO'] - opponent_elo)
        matches = matches.sort_values('EloDiff').head(n)

        gf = matches[gf_col].mean() if not matches.empty else 1.0
        ga = matches[ga_col].mean() if not matches.empty else 1.0
        return gf, ga

    elo_away = elo_dict.get(away_team, 1500)
    elo_home = elo_dict.get(home_team, 1500)

    hist_home, hist_home_ga = filter_by_elo(df_hist, home_team, True, elo_away)
    hist_away, hist_away_ga = filter_by_elo(df_hist, away_team, False, elo_home)

    season_home, season_home_ga = filter_by_elo(df_season, home_team, True, elo_away)
    season_away, season_away_ga = filter_by_elo(df_season, away_team, False, elo_home)

    last5_home, last5_home_ga = filter_by_elo(df_last10, home_team, True, elo_away)
    last5_away, last5_away_ga = filter_by_elo(df_last10, away_team, False, elo_home)

    league_avg_home_goals = df['FTHG'].mean()
    league_avg_away_goals = df['FTAG'].mean()

    def compute_expected(gf, ga, l_home, l_away):
        return l_home * (gf / l_home) * (ga / l_away)

    ehist_home = compute_expected(hist_home, hist_away_ga, league_avg_home_goals, league_avg_away_goals)
    eseason_home = compute_expected(season_home, season_away_ga, league_avg_home_goals, league_avg_away_goals)
    elast5_home = compute_expected(last5_home, last5_away_ga, league_avg_home_goals, league_avg_away_goals)

    ehist_away = compute_expected(hist_away, hist_home_ga, league_avg_away_goals, league_avg_home_goals)
    eseason_away = compute_expected(season_away, season_home_ga, league_avg_away_goals, league_avg_home_goals)
    elast5_away = compute_expected(last5_away, last5_home_ga, league_avg_away_goals, league_avg_home_goals)

    base_expected_home = 0.3 * ehist_home + 0.4 * eseason_home + 0.3 * elast5_home
    base_expected_away = 0.3 * ehist_away + 0.4 * eseason_away + 0.3 * elast5_away

    final_home_advantage = calculate_team_home_advantage(df, home_team)

    expected_home = base_expected_home + final_home_advantage
    expected_away = base_expected_away

    return round(expected_home, 2), round(expected_away, 2)

def expected_goals_vs_opponent_strength_weighted(df, team, opponent, elo_dict, is_home=True, n=20):
    """
    Vypočítá očekávané góly na základě toho, jak tým skóruje proti soupeřům podobné síly jako aktuální soupeř.
    Síla soupeřů se určuje podle ELO ratingu (rozděleno do tříd: strong, average, weak).
    """
    df = prepare_df(df)
    df = df.sort_values("Date")

    # Filtruj posledních N zápasů týmu v dané roli
    team_matches = df[df['HomeTeam'] == team] if is_home else df[df['AwayTeam'] == team]
    team_matches = team_matches.tail(n)

    if team_matches.empty:
        return 1.0  # fallback

    # Nastav sloupce podle role
    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    opp_col = 'AwayTeam' if is_home else 'HomeTeam'
    gf_col = 'FTHG' if is_home else 'FTAG'
    ga_col = 'FTAG' if is_home else 'FTHG'

    # Přidej ELO soupeřů a klasifikaci
    team_matches = team_matches.copy()
    team_matches["Opponent"] = team_matches[opp_col]
    team_matches["EloOpp"] = team_matches["Opponent"].map(elo_dict)
    team_matches = team_matches.dropna(subset=["EloOpp"])

    opp_elo = elo_dict.get(opponent, 1500)

    # Percentilové rozdělení soupeřů v lize
    all_elos = list(elo_dict.values())
    p30 = np.percentile(all_elos, 30)
    p70 = np.percentile(all_elos, 70)

    def classify(e):
        if e <= p30:
            return "weak"
        elif e >= p70:
            return "strong"
        else:
            return "average"

    team_matches["OppStrength"] = team_matches["EloOpp"].apply(classify)
    current_strength = classify(opp_elo)

    # Výpočet průměrů podle třídy soupeřů
    gfs = {}
    for group in ["strong", "average", "weak"]:
        sub = team_matches[team_matches["OppStrength"] == group]
        gfs[group] = sub[gf_col].mean() if not sub.empty else 1.0

    # Váhy dle podobnosti aktuálního soupeře
    weights = {
        "strong": 0.6 if current_strength == "strong" else 0.2,
        "average": 0.6 if current_strength == "average" else 0.2,
        "weak": 0.6 if current_strength == "weak" else 0.2
    }

    # Vážený průměr
    expected = (
        weights["strong"] * gfs["strong"] +
        weights["average"] * gfs["average"] +
        weights["weak"] * gfs["weak"]
    )

    return round(expected, 2)

def expected_goals_combined_homeaway_allmatches(
    df, home_team, away_team, elo_dict,
    weight_homeaway=0.4,
    weight_all=0.3,
    weight_matchup=0.3
):
    df = prepare_df(df)
    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)

    df_hist = df[df['Date'] < one_year_ago]
    df_season = df[df['Date'] >= one_year_ago]

    # posledních 5 zápasů
    df_last5_home = get_last_n_matches(df, home_team, role="home")
    df_last5_away = get_last_n_matches(df, away_team, role="away")
    df_last5_all_home = get_last_n_matches(df, home_team)
    df_last5_all_away = get_last_n_matches(df, away_team)

    league_avg_home = df['FTHG'].mean()
    league_avg_away = df['FTAG'].mean()

    def safe_stat(series, default=1.0):
        val = series.mean()
        return val if not np.isnan(val) else default

    def get_home_away_exp(sub, team, is_home):
        df_team = sub[sub['HomeTeam'] == team] if is_home else sub[sub['AwayTeam'] == team]
        gf = safe_stat(df_team['FTHG'] if is_home else df_team['FTAG'])
        ga = safe_stat(df_team['FTAG'] if is_home else df_team['FTHG'])
        return gf, ga

    def get_all_matches_exp(sub, team):
        matches = sub[(sub['HomeTeam'] == team) | (sub['AwayTeam'] == team)]
        gf, ga = [], []
        for _, row in matches.iterrows():
            if row['HomeTeam'] == team:
                gf.append(row['FTHG'])
                ga.append(row['FTAG'])
            else:
                gf.append(row['FTAG'])
                ga.append(row['FTHG'])
        return safe_stat(pd.Series(gf)), safe_stat(pd.Series(ga))

    def compute_expected(gf, ga_opp):
        return league_avg_home * (gf / league_avg_home) * (ga_opp / league_avg_away)

    def compute_weighted(dfh_list, dfa_list, extractor_home, extractor_away):
        e_home, e_away = [], []
        for dfh, dfa in zip(dfh_list, dfa_list):
            gf_h, ga_h = extractor_home(dfh)
            gf_a, ga_a = extractor_away(dfa)
            e_home.append(compute_expected(gf_h, ga_a))
            e_away.append(compute_expected(gf_a, ga_h))
        w_home = 0.15 * e_home[0] + 0.5 * e_home[1] + 0.35 * e_home[2]
        w_away = 0.15 * e_away[0] + 0.5 * e_away[1] + 0.35 * e_away[2]
        return w_home, w_away, e_home, e_away

    # 1. Home/Away přístup
    ha_home, ha_away, ha_parts_home, ha_parts_away = compute_weighted(
        [df_hist, df_season, df_last5_home],
        [df_hist, df_season, df_last5_away],
        lambda d: get_home_away_exp(d, home_team, True),
        lambda d: get_home_away_exp(d, away_team, False)
    )

    # 2. All matches přístup
    all_home, all_away, all_parts_home, all_parts_away = compute_weighted(
        [df_hist, df_season, df_last5_all_home],
        [df_hist, df_season, df_last5_all_away],
        lambda d: get_all_matches_exp(d, home_team),
        lambda d: get_all_matches_exp(d, away_team)
    )

    # 3. Matchup přístup (proti silným/průměrným/slabým soupeřům)
    matchup_home = expected_goals_vs_opponent_strength_weighted(df, home_team, away_team, elo_dict, is_home=True)
    matchup_away = expected_goals_vs_opponent_strength_weighted(df, away_team, home_team, elo_dict, is_home=False)

    # Kombinace všech tří přístupů
    final_home = round(
        weight_homeaway * ha_home +
        weight_all * all_home +
        weight_matchup * matchup_home, 2)

    final_away = round(
        weight_homeaway * ha_away +
        weight_all * all_away +
        weight_matchup * matchup_away, 2)

    # Debug výstup
    def print_parts(label, parts_home, parts_away, combined_home, combined_away):
        print(f"{label}")
        print(f"  Historie:      {poisson_over25_probability(parts_home[0], parts_away[0])}%")
        print(f"  Sezóna:        {poisson_over25_probability(parts_home[1], parts_away[1])}%")
        print(f"  Posledních 5:  {poisson_over25_probability(parts_home[2], parts_away[2])}%")
        print(f"  => Průměr:     {poisson_over25_probability(combined_home, combined_away)}%")

    print_parts("🟦 Home/Away-only přístup – Over 2.5:", ha_parts_home, ha_parts_away, ha_home, ha_away)
    print_parts("🟧 All matches přístup – Over 2.5:", all_parts_home, all_parts_away, all_home, all_away)
    print("🎯 Matchup přístup (síla soupeře):")
    print(f"  Home exp:      {matchup_home}  |  Away exp: {matchup_away}")
    print(f"  Over 2.5:      {poisson_over25_probability(matchup_home, matchup_away)}%")
    print("✅ Finální kombinovaná Over 2.5:")
    print(f"  Výsledek:      {poisson_over25_probability(final_home, final_away)}%")

    return final_home, final_away

import pandas as pd

TEAM_COMPARISON_ICON_MAP = {
    "Góly": "⚽",
    "Obdržené góly": "🥅",
    "Střely": "📸",
    "Na branku": "🎯",
    "Rohy": "🚩",
    "Fauly": "⚠️",
    "Žluté": "🟨",
    "Červené": "🟥",
    "Ofenzivní efektivita": "⚡",
    "Defenzivní efektivita": "🛡️",
    "Přesnost střel": "🎯",
    "Konverzní míra": "🌟",
    "Čistá konta %": "🧤",
    "Over 2.5 %": "📈",
    "BTTS %": "🎯",
}

TEAM_COMPARISON_DESC_MAP = {
    "Góly": "Průměr vstřelených gólů na zápas",
    "Obdržené góly": "Průměr inkasovaných gólů na zápas",
    "Střely": "Průměr střel na zápas",
    "Na branku": "Střely mířící na branku",
    "Rohy": "Počet rozehraných rohů",
    "Fauly": "Počet faulů",
    "Žluté": "Žluté karty",
    "Červené": "Červené karty",
    "Ofenzivní efektivita": "Střely potřebné na gól (nižší je lepší)",
    "Defenzivní efektivita": "Inkasované góly na střelu soupeře (nižší je lepší)",
    "Přesnost střel": "Podíl střel na branku v %",
    "Konverzní míra": "Podíl gólů ze střel v %",
    "Čistá konta %": "Podíl zápasů bez obdrženého gólu",
    "Over 2.5 %": "Zápasy s více než 2.5 góly",
    "BTTS %": "Zápasy, kde skórovaly oba týmy",
}

# Určuje, zda je u dané metriky výhodnější vyšší hodnota (True), nebo nižší hodnota (False)
TEAM_COMPARISON_HIGHER_IS_BETTER = {
    "Góly": True,
    "Obdržené góly": False,
    "Střely": True,
    "Na branku": True,
    "Rohy": True,
    "Fauly": False,
    "Žluté": False,
    "Červené": False,
    "Ofenzivní efektivita": False,
    "Defenzivní efektivita": False,
    "Přesnost střel": True,
    "Konverzní míra": True,
    "Čistá konta %": True,
    "Over 2.5 %": True,
    "BTTS %": True,
}


def render_team_comparison_section(team1, team2, stats_total, stats_home, stats_away):
    st.markdown(f"## 🆚 Porovnání týmů: {team1} vs {team2}")
    metrics = list(TEAM_COMPARISON_ICON_MAP.keys())

    with st.expander("Legenda"):
        for met in metrics:
            icon = TEAM_COMPARISON_ICON_MAP.get(met, "")
            desc = TEAM_COMPARISON_DESC_MAP.get(met, "")
            st.markdown(f"{icon} {met} - {desc}")

    col_celkem, col_doma, col_venku = st.columns(3)

    def _build_table(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for met in metrics:
            if met not in df.index:
                continue
            icon = TEAM_COMPARISON_ICON_MAP.get(met, "")
            try:
                v1 = float(df.at[met, "team1"])
                v2 = float(df.at[met, "team2"])
            except KeyError:
                continue
            higher_better = TEAM_COMPARISON_HIGHER_IS_BETTER.get(met, True)
            if v1 == v2:
                better = "="
            else:
                if higher_better:
                    better = team1 if v1 > v2 else team2
                else:
                    better = team1 if v1 < v2 else team2
            rows.append({
                "Metrika": f"{icon} {met}",
                "team1": round(v1, 2),
                "team2": round(v2, 2),
                "Lepší": better,
            })
        return pd.DataFrame(rows, columns=["Metrika", "team1", "team2", "Lepší"])

    def _style_and_display(df: pd.DataFrame):
        legend_html = (
            f"<span style='background-color:#add8e6;padding:0 8px;border-radius:4px;'>&nbsp;</span> {team1}"
            f" &nbsp; <span style='background-color:#d3d3d3;padding:0 8px;border-radius:4px;'>&nbsp;</span> {team2}"
        )
        st.caption(legend_html, unsafe_allow_html=True)
        def _highlight(row):
            met = row["Metrika"].split(" ", 1)[1]
            higher_better = TEAM_COMPARISON_HIGHER_IS_BETTER.get(met, True)
            v1, v2 = row["team1"], row["team2"]
            color1 = color2 = ""
            if higher_better:
                if v1 > v2:
                    color1 = "background-color: lightgreen"
                elif v2 > v1:
                    color2 = "background-color: lightgreen"
            else:
                if v1 < v2:
                    color1 = "background-color: lightgreen"
                elif v2 < v1:
                    color2 = "background-color: lightgreen"
            return pd.Series([color1, color2], index=["team1", "team2"])

        styled = (
            df.style
            .set_properties(subset=["team1"], **{"background-color": "#add8e6"})
            .set_properties(subset=["team2"], **{"background-color": "#d3d3d3"})
            .apply(_highlight, axis=1, subset=["team1", "team2"])
        )
        st.dataframe(
            styled,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Metrika": "Metrika",
                "team1": st.column_config.NumberColumn(team1),
                "team2": st.column_config.NumberColumn(team2),
                "Lepší": "Lepší",
            },
        )

    with col_celkem:
        st.markdown("### Celkem")
        _style_and_display(_build_table(stats_total))
    with col_doma:
        st.markdown("### 🏠 Doma")
        _style_and_display(_build_table(stats_home))
    with col_venku:
        st.markdown("### 🚌 Venku")
        _style_and_display(_build_table(stats_away))




def generate_team_comparison(df: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
    def team_stats(df, team):
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]

        matches = pd.concat([home, away])
        if matches.empty:
            return {}

        goals = pd.concat([home['FTHG'], away['FTAG']]).mean()
        goals_conceded = pd.concat([home['FTAG'], away['FTHG']]).mean()
        shots = pd.concat([home['HS'], away['AS']]).mean()
        shots_on_target = pd.concat([home['HST'], away['AST']]).mean()
        corners = pd.concat([home['HC'], away['AC']]).mean()
        fouls = pd.concat([home['HF'], away['AF']]).mean()
        yellows = pd.concat([home['HY'], away['AY']]).mean()
        reds = pd.concat([home['HR'], away['AR']]).mean()

        offensive_eff = shots / goals if goals else 0
        defensive_eff = goals_conceded / shots if shots else 0
        accuracy = shots_on_target / shots if shots else 0
        conversion = goals / shots if shots else 0

        clean_sheets = 0
        total_matches = 0
        over25 = 0
        btts = 0

        for _, row in matches.iterrows():
            is_home = row['HomeTeam'] == team
            gf = row['FTHG'] if is_home else row['FTAG']
            ga = row['FTAG'] if is_home else row['FTHG']
            if ga == 0:
                clean_sheets += 1
            if row['FTHG'] + row['FTAG'] > 2.5:
                over25 += 1
            if row['FTHG'] > 0 and row['FTAG'] > 0:
                btts += 1
            total_matches += 1

        return {
            "Góly": goals,
            "Obdržené góly": goals_conceded,
            "Střely": shots,
            "Na branku": shots_on_target,
            "Rohy": corners,
            "Fauly": fouls,
            "Žluté": yellows,
            "Červené": reds,
            "Ofenzivní efektivita": offensive_eff,
            "Defenzivní efektivita": defensive_eff,
            "Přesnost střel": accuracy * 100,
            "Konverzní míra": conversion * 100,
            "Čistá konta %": (clean_sheets / total_matches) * 100 if total_matches else 0,
            "Over 2.5 %": (over25 / total_matches) * 100 if total_matches else 0,
            "BTTS %": (btts / total_matches) * 100 if total_matches else 0,
        }

    stats1 = team_stats(df, team1)
    stats2 = team_stats(df, team2)

    metrics = sorted(set(stats1.keys()) | set(stats2.keys()))
    rows = []
    for m in metrics:
        val1 = stats1.get(m, 0)
        val2 = stats2.get(m, 0)
        rows.append([m, round(val1, 1), round(val2, 1)])

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=["Metrika", "team1", "team2"]).set_index("Metrika")



