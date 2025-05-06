import pandas as pd
import numpy as np

from .core import prepare_df, calculate_points
from .xg import calculate_team_pseudo_xg
from utils.utils_warnings import detect_overperformance_and_momentum

def calculate_clean_sheets(df: pd.DataFrame, team: str) -> float:
    """Vrací procento zápasů, kdy tým udržel čisté konto."""
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    cs = 0
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team and row['FTAG'] == 0:
            cs += 1
        elif row['AwayTeam'] == team and row['FTHG'] == 0:
            cs += 1
    return round(100 * cs / len(team_matches), 1) if len(team_matches) > 0 else 0

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
    from .core import prepare_df, calculate_points

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

