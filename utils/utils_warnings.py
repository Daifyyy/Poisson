
import pandas as pd
import numpy as np
from scipy.stats import poisson
import streamlit as st
from itertools import product

from utils.poisson_utils.stats import calculate_points

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['FTHG', 'FTAG', 'Date'])
    df = df.sort_values('Date')
    return df

def prepare_df(df):
    """Základní úprava dat: kopírování, převod datumu, odstranění nevalidních řádků, seřazení podle data."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    return df

def calculate_team_goal_averages(df):
    """Vrátí slovník: tým -> průměr vstřelených gólů (kombinace doma a venku)."""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    avg_goals = {}
    for team in teams:
        home_avg = df[df['HomeTeam'] == team]['FTHG'].mean()
        away_avg = df[df['AwayTeam'] == team]['FTAG'].mean()
        avg_goals[team] = np.nanmean([home_avg, away_avg])
    return avg_goals

def form_points_to_emoji(avg_points):
    """Vrací emoji reprezentaci podle průměrného počtu bodů na zápas."""
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



def split_teams_by_strength(df):
    """Rozdělí týmy do kategorií podle průměrného počtu vstřelených gólů."""
    df = df.copy()
    avg_goals = calculate_team_goal_averages(df)


    sorted_teams = sorted(avg_goals.items(), key=lambda x: x[1], reverse=True)
    total = len(sorted_teams)
    top_teams = set(t for t, _ in sorted_teams[:int(total * 0.3)])
    bottom_teams = set(t for t, _ in sorted_teams[-int(total * 0.3):])
    middle_teams = set(avg_goals.keys()) - top_teams - bottom_teams

    return top_teams, middle_teams, bottom_teams



def detect_current_season(df):
    df = prepare_df(df)

    dates = df['Date'].drop_duplicates().sort_values().reset_index(drop=True)
    date_diffs = dates.diff().fillna(pd.Timedelta(days=0))
    season_breaks = dates[date_diffs > pd.Timedelta(days=30)].reset_index(drop=True)

    if not season_breaks.empty:
        season_start = season_breaks.iloc[-1]  # poslední pauza => začátek aktuální sezony
    else:
        season_start = dates.iloc[0]  # fallback – bereme vše
    print(season_start)
    return df[df['Date'] >= season_start], season_start

def calculate_team_strengths(df):
    league_avg_home_goals = df['FTHG'].mean()
    league_avg_away_goals = df['FTAG'].mean()
    teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    strengths = []
    for team in teams:
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]
        attack_home = home_matches['FTHG'].mean() / league_avg_home_goals if not home_matches.empty else 1
        attack_away = away_matches['FTAG'].mean() / league_avg_away_goals if not away_matches.empty else 1
        defense_home = home_matches['FTAG'].mean() / league_avg_away_goals if not home_matches.empty else 1
        defense_away = away_matches['FTHG'].mean() / league_avg_home_goals if not away_matches.empty else 1
        strengths.append({
            'Team': team,
            'AttackHome': round(attack_home, 3),
            'AttackAway': round(attack_away, 3),
            'DefenseHome': round(defense_home, 3),
            'DefenseAway': round(defense_away, 3)
        })
    return pd.DataFrame(strengths), league_avg_home_goals, league_avg_away_goals

def add_btts_column(df):
    """Přidá sloupec 'BTTS' indikující, zda oba týmy skórovaly."""
    df = df.copy()
    df['BTTS'] = df.apply(lambda row: int(row['FTHG'] > 0 and row['FTAG'] > 0), axis=1)
    return df

# def aggregate_team_stats(df):
#     """Agreguje statistiky za všechny zápasy (doma i venku) pro každý tým."""
#     teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
#     records = []

#     for team in teams:
#         home = df[df['HomeTeam'] == team]
#         away = df[df['AwayTeam'] == team]

#         goals = pd.concat([home['FTHG'], away['FTAG']])
#         conceded = pd.concat([home['FTAG'], away['FTHG']])
#         shots = pd.concat([home['HS'], away['AS']])
#         shots_on_target = pd.concat([home['HST'], away['AST']])
#         corners = pd.concat([home['HC'], away['AC']])
#         yellows = pd.concat([home['HY'], away['AY']])

#         records.append({
#             "Tým": team,
#             "Góly": goals.mean(),
#             "Obdržené góly": conceded.mean(),
#             "Střely": shots.mean(),
#             "Na branku": shots_on_target.mean(),
#             "Rohy": corners.mean(),
#             "Žluté": yellows.mean()
#         })

#     df_stats = pd.DataFrame(records).set_index("Tým")

#     print(df_stats.columns)
#     return df_stats



def calculate_conceded_goals(df):
    """Vrací DataFrame s průměrným počtem obdržených gólů pro každý tým."""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

    conceded_stats = []
    for team in teams:
        home = df[df['AwayTeam'] == team]
        away = df[df['HomeTeam'] == team]
        goals_against = pd.concat([home['FTHG'], away['FTAG']]).mean()
        conceded_stats.append({"Tým": team, "Obdržené góly": round(goals_against, 2)})

    return pd.DataFrame(conceded_stats).sort_values("Obdržené góly", ascending=False).reset_index(drop=True)

def calculate_recent_team_form(df, last_n=5):
    """Vrací DataFrame s průměrem bodů a formou (emoji) za posledních N zápasů pro každý tým."""
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


def calculate_form_emojis(df, days=31):
    """Vrací dictionary: tým -> emoji reprezentace formy."""
    form_dict = calculate_recent_form(df, days=days)
    form_emojis = {}

    for team, avg_points in form_dict.items():
        form_emojis[team] = form_points_to_emoji(avg_points)

    return form_emojis


def poisson_prediction(home_exp, away_exp, max_goals=6):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i][j] = poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)
    return matrix

def match_outcomes_prob(matrix):
    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    return {'Home Win': round(home_win * 100, 2), 'Draw': round(draw * 100, 2), 'Away Win': round(away_win * 100, 2)}

def over_under_prob(matrix, line=2.5):
    prob_over = sum([matrix[i][j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if i + j > line])
    return {'Over 2.5': round(prob_over * 100, 2), 'Under 2.5': round((1 - prob_over) * 100, 2)}

def btts_prob(matrix):
    btts = sum([matrix[i][j] for i in range(1, matrix.shape[0]) for j in range(1, matrix.shape[1])])
    return {'BTTS Yes': round(btts * 100, 2), 'BTTS No': round((1 - btts) * 100, 2)}

def prob_to_odds(prob_percent):
    if prob_percent <= 0:
        return "∞"
    return round(100 / prob_percent, 2)

def generate_score_table_df(matrix, home_team, away_team):
    df = pd.DataFrame(matrix * 100)
    df.index.name = f"{home_team} góly"
    df.columns.name = f"{away_team} góly"

    styled = df.style\
        .format("{:.1f} %")\
        .background_gradient(cmap="YlOrRd", axis=None)\
        .set_properties(**{
            "text-align": "center",
            "font-size": "11px"
        })

    return styled


def get_top_scorelines(matrix, top_n=5):
    score_probs = [((i, j), matrix[i][j]) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    score_probs.sort(key=lambda x: x[1], reverse=True)
    return score_probs[:top_n]

def expected_team_stats_weighted_by_elo(df, home_team, away_team, stat_columns, elo_dict):
    df = prepare_df(df)


    # Získání ELO soupeře
    elo_away = elo_dict.get(away_team, 1500)
    elo_home = elo_dict.get(home_team, 1500)

    # Vypočítání percentilů ELO
    all_elos = list(elo_dict.values())
    high_threshold = np.percentile(all_elos, 70)
    low_threshold = np.percentile(all_elos, 30)

    def classify_opponent(elo):
        if elo >= high_threshold:
            return 'strong'
        elif elo <= low_threshold:
            return 'weak'
        else:
            return 'average'

    def filter_matches(team, is_home, opponent_elo):
        team_col = 'HomeTeam' if is_home else 'AwayTeam'
        opp_col = 'AwayTeam' if is_home else 'HomeTeam'
        matches = df[df[team_col] == team]
        matches['OppELO'] = matches[opp_col].map(elo_dict)
        matches['EloDiff'] = abs(matches['OppELO'] - opponent_elo)
        return matches.sort_values('EloDiff').head(10)  # top 10 nejbližších podle ELO

    output = {}
    for stat_name, (home_col, away_col) in stat_columns.items():
        if home_col not in df.columns or away_col not in df.columns:
            output[stat_name] = {'Home': 0.0, 'Away': 0.0}
            continue

        home_matches = filter_matches(home_team, is_home=True, opponent_elo=elo_away)
        away_matches = filter_matches(away_team, is_home=False, opponent_elo=elo_home)

        home_vals = []
        for _, row in home_matches.iterrows():
            if row['HomeTeam'] == home_team:
                val = row.get(home_col, 0)
            else:
                val = row.get(away_col, 0)
            home_vals.append(val)

        away_vals = []
        for _, row in away_matches.iterrows():
            if row['AwayTeam'] == away_team:
                val = row.get(away_col, 0)
            else:
                val = row.get(home_col, 0)
            away_vals.append(val)

        output[stat_name] = {
            'Home': round(np.mean(home_vals), 1) if home_vals else 0.0,
            'Away': round(np.mean(away_vals), 1) if away_vals else 0.0
        }

    return output



def expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict):
    df = prepare_df(df)


    # Rozdělení období
    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)
    df_hist = df[df['Date'] < one_year_ago]
    df_season = df[df['Date'] >= one_year_ago]
    df_last5 = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team) |
                  (df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].sort_values('Date').tail(10)

    # Pomocná funkce na základě ELO
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

    # Všechny tři složky
    hist_home, hist_home_ga = filter_by_elo(df_hist, home_team, is_home=True, opponent_elo=elo_away)
    hist_away, hist_away_ga = filter_by_elo(df_hist, away_team, is_home=False, opponent_elo=elo_home)

    season_home, season_home_ga = filter_by_elo(df_season, home_team, is_home=True, opponent_elo=elo_away)
    season_away, season_away_ga = filter_by_elo(df_season, away_team, is_home=False, opponent_elo=elo_home)

    last5_home, last5_home_ga = filter_by_elo(df_last5, home_team, is_home=True, opponent_elo=elo_away)
    last5_away, last5_away_ga = filter_by_elo(df_last5, away_team, is_home=False, opponent_elo=elo_home)

    # Ligové průměry
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

    expected_home = 0.3 * ehist_home + 0.4 * eseason_home + 0.3 * elast5_home
    expected_away = 0.3 * ehist_away + 0.4 * eseason_away + 0.3 * elast5_away

    return round(expected_home, 2), round(expected_away, 2)

def calculate_pseudo_xg_for_team(df, team):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    # Filtrování dat pro daný tým
    home_matches = df[df['HomeTeam'] == team]
    away_matches = df[df['AwayTeam'] == team]

    # Koeficienty pro různé typy střel (xG model)
    shot_coeffs = {
        "on_target": 0.1,  # Střela na branku
        "off_target": 0.05,  # Střela mimo branku
        "long_distance": 0.05,  # Střela z dálky
        "inside_box": 0.7  # Střela z pokutového území
    }

    # Počítáme vážený xG pro domácí a venkovní zápasy
    home_xg = (home_matches['HST'] * shot_coeffs["on_target"] + home_matches['HS'] * shot_coeffs["off_target"]).mean()
    away_xg = (away_matches['AST'] * shot_coeffs["on_target"] + away_matches['AS'] * shot_coeffs["off_target"]).mean()

    total_shots = df['HS'].where(df['HomeTeam'] == team, df['AS'])
    total_sot = df['HST'].where(df['HomeTeam'] == team, df['AST'])
    total_goals = df['FTHG'].where(df['HomeTeam'] == team, df['FTAG'])

    # Vypočítáme celkové xG pro tým
    xg_total = (total_sot * shot_coeffs["on_target"] + total_shots * shot_coeffs["off_target"]).sum()
    avg_xg = xg_total / len(df) if len(df) > 0 else 0

    # Počítání gólů a inkasovaných gólů
    total_goals_sum = total_goals.sum()
    goals_home = home_matches['FTHG'].sum()
    goals_away = away_matches['FTAG'].sum()
    conceded_home = home_matches['FTAG'].sum()
    conceded_away = away_matches['FTHG'].sum()

    return {
        "avg_xG": round(avg_xg, 3),
        "xG_home": round(home_xg, 3),
        "xG_away": round(away_xg, 3),
        "xG_per_goal": round(xg_total / total_goals_sum, 2) if total_goals_sum > 0 else 0,
        "xG_total": round(xg_total, 2),
        "goals_home": goals_home,
        "goals_away": goals_away,
        "conceded_home": conceded_home,
        "conceded_away": conceded_away,
        "xG_total_home": round(home_xg, 2),
        "xG_total_away": round(away_xg, 2)
    }

def calculate_team_pseudo_xg(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    results = {}
    for team in teams:
        results[team] = calculate_pseudo_xg_for_team(df, team)
    return results

def calculate_expected_points(outcomes: dict) -> dict:
    """Calculate expected points based on outcome probabilities."""
    home_xp = (outcomes['Home Win'] / 100) * 3 + (outcomes['Draw'] / 100) * 1
    away_xp = (outcomes['Away Win'] / 100) * 3 + (outcomes['Draw'] / 100) * 1
    return {
        'Home xP': round(home_xp, 1),
        'Away xP': round(away_xp, 1)
    }
    
def analyze_opponent_strength(df, team, is_home=True):
    df = prepare_df(df)


    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    opp_col = 'AwayTeam' if is_home else 'HomeTeam'
    goals_col = 'FTHG' if is_home else 'FTAG'
    shots_col = 'HS' if is_home else 'AS'

    team_matches = df[df[team_col] == team]

    # Nové rozdělení týmů
    top_teams, middle_teams, bottom_teams = split_teams_by_strength(df)

    performance = {
        'strong': [],
        'average': [],
        'weak': []
    }

    for _, row in team_matches.iterrows():
        opponent = row[opp_col]
        goals = row[goals_col]
        shots = row[shots_col]
        points = 3 if row['FTHG'] > row['FTAG'] and is_home else 3 if row['FTAG'] > row['FTHG'] and not is_home else 1 if row['FTHG'] == row['FTAG'] else 0

        data_point = {
            'goals': goals,
            'shots': shots,
            'points': points
        }

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

    
def calculate_elo_ratings(df):
    df = prepare_df(df)


    elo = {}
    k = 20
    base_elo = 1500

    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']

        elo_home = elo.get(home, base_elo)
        elo_away = elo.get(away, base_elo)

        expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
        expected_away = 1 - expected_home

        if home_goals > away_goals:
            score_home, score_away = 1, 0
        elif home_goals < away_goals:
            score_home, score_away = 0, 1
        else:
            score_home = score_away = 0.5

        elo[home] = elo_home + k * (score_home - expected_home)
        elo[away] = elo_away + k * (score_away - expected_away)

    return {team: round(score, 1) for team, score in elo.items()}

def calculate_elo_changes(df):
    """Vrací DataFrame změn ELO ratingů mezi začátkem a koncem sezóny."""
    df = prepare_df(df)
    start_date = df['Date'].min() + pd.Timedelta(days=5)

    elo_start = calculate_elo_ratings(df[df['Date'] <= start_date])
    elo_end = calculate_elo_ratings(df)

    elo_change = [{"Tým": t, "Změna": round(elo_end.get(t, 1500) - elo_start.get(t, 1500), 1)} for t in elo_end]

    return pd.DataFrame(elo_change).sort_values("Změna", ascending=False).reset_index(drop=True)



def calculate_recent_form(df, days=30):
    df = prepare_df(df)

    recent_df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=days)]
    team_points = {}
    team_matches = recent_df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]

    for _, row in team_matches.iterrows():
        home, away, fthg, ftag = row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG']

        if fthg > ftag:
            team_points[home] = team_points.get(home, []) + [3]
            team_points[away] = team_points.get(away, []) + [0]
        elif fthg < ftag:
            team_points[home] = team_points.get(home, []) + [0]
            team_points[away] = team_points.get(away, []) + [3]
        else:
            team_points[home] = team_points.get(home, []) + [1]
            team_points[away] = team_points.get(away, []) + [1]

    return {team: round(np.mean(points), 2) for team, points in team_points.items() if len(points) > 0}

def calculate_expected_and_actual_points(df):
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

def extended_home_away_opponent_form(df, team):
    df = prepare_df(df)

    # Rozdělení podle síly soupeřů
    top_teams, middle_teams, bottom_teams = split_teams_by_strength(df)


    def summarize(matches):
        if matches.empty:
            return {"g": 0, "s": 0, "sot": 0, "pts": 0, "xg": 0, "games": 0}
        g = matches["goals"].mean()
        s = matches["shots"].mean()
        sot = matches["sot"].mean()
        conv = round(g / sot, 2) if sot > 0 else 0
        xg = round(sot * conv, 2)
        pts = matches["points"].mean()
        return {"g": round(g, 2), "s": round(s, 1), "sot": round(sot, 1), "pts": round(pts, 2), "xg": xg, "games": len(matches)}

    result = {"home": {}, "away": {}}
    for loc, is_home in [("home", True), ("away", False)]:
        col_team = 'HomeTeam' if is_home else 'AwayTeam'
        col_opp = 'AwayTeam' if is_home else 'HomeTeam'
        goals_col = 'FTHG' if is_home else 'FTAG'
        shots_col = 'HS' if is_home else 'AS'
        sot_col = 'HST' if is_home else 'AST'

        subset = df[df[col_team] == team].copy()
        subset["opponent"] = subset[col_opp]
        subset["goals"] = subset[goals_col]
        subset["shots"] = subset[shots_col]
        subset["sot"] = subset[sot_col]
        subset["points"] = subset.apply(lambda r: calculate_points(r, is_home), axis=1)

        result[loc]["vs_strong"] = summarize(subset[subset["opponent"].isin(top_teams)])
        result[loc]["vs_average"] = summarize(subset[subset["opponent"].isin(middle_teams)])
        result[loc]["vs_weak"] = summarize(subset[subset["opponent"].isin(bottom_teams)])
        
    return result

def merged_home_away_opponent_form(df, team):
    df = prepare_df(df)


    # Nové rozdělení týmů podle síly
    top_teams, middle_teams, bottom_teams = split_teams_by_strength(df)


    def summarize(matches, is_home):
        if matches.empty:
            return {"Z": 0, "G": 0, "OG": 0, "S": 0, "SOT": 0, "xG": 0, "PTS": 0, "CS": 0}
        goals_for = matches["FTHG"] if is_home else matches["FTAG"]
        goals_against = matches["FTAG"] if is_home else matches["FTHG"]
        shots = matches["HS"] if is_home else matches["AS"]
        sot = matches["HST"] if is_home else matches["AST"]
        conv = goals_for.mean() / sot.mean() if sot.mean() > 0 else 0
        xg = round(sot.mean() * conv, 2)
        points = matches.apply(lambda r: calculate_points(r, is_home), axis=1).mean()
        clean_sheets = (goals_against == 0).sum()
        cs_percent = round(100 * clean_sheets / len(matches), 1)
        return {
            "Z": len(matches),
            "G": round(goals_for.mean(), 2),
            "OG": round(goals_against.mean(), 2),
            "S": round(shots.mean(), 1),
            "SOT": round(sot.mean(), 1),
            "xG": xg,
            "PTS": round(points, 2),
            "CS": cs_percent
        }

    def generate_table():
        result = {}
        for label, group in [("💪 Silní", top_teams), ("⚖️ Průměrní", middle_teams), ("🪶 Slabí", bottom_teams)]:
            home = df[(df['HomeTeam'] == team) & (df['AwayTeam'].isin(group))]
            away = df[(df['AwayTeam'] == team) & (df['HomeTeam'].isin(group))]
            home_stats = summarize(home, is_home=True)
            away_stats = summarize(away, is_home=False)
            result[label] = {
                "Zápasy": f"{home_stats['Z']} / {away_stats['Z']}",
                "Góly": f"{home_stats['G']} / {away_stats['G']}",
                "Obdržené": f"{home_stats['OG']} / {away_stats['OG']}",
                "Střely": f"{home_stats['S']} / {away_stats['S']}",
                "Na branku": f"{home_stats['SOT']} / {away_stats['SOT']}",
                "xG": f"{home_stats['xG']} / {away_stats['xG']}",
                "Body/zápas": f"{home_stats['PTS']} / {away_stats['PTS']}",
                "Čistá konta %": f"{home_stats['CS']} / {away_stats['CS']}"
            }
        return result

    return generate_table()


def get_head_to_head_stats(df, home_team, away_team, last_n=5):
    h2h = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
             ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))].sort_values('Date', ascending=False).head(last_n)

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

def calculate_match_tempo(df, team, opponent_elo, is_home, elo_dict, last_n=10):
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
            "aggressiveness_rating": "N/A"
        }

    # Základní složky pro tempo
    shots = matches['HS'] if is_home else matches['AS']
    corners = matches['HC'] if is_home else matches['AC']
    fouls = matches['HF'] if is_home else matches['AF']
    tempo_index = (shots + corners + fouls).mean()

    # 💥 Výpočet Aggressiveness Indexu
    yellow_cards = matches['HY'] if is_home else matches['AY']
    red_cards = matches['HR'] if is_home else matches['AR']
    aggressiveness_index = ((yellow_cards + 2 * red_cards + fouls).sum()) / len(matches)

    # Percentil ligového tempa
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

    # Dominance vs trpění
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

    # 🎭 Emoji hodnocení tvrdosti
    if aggressiveness_index < 14:
        aggressiveness_rating = "🕊️ velmi klidné"
    elif aggressiveness_index < 19:
        aggressiveness_rating = "🟢 korektní"
    elif aggressiveness_index < 24:
        aggressiveness_rating = "🟡 tvrdší zápas"
    elif aggressiveness_index < 30:
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
        "aggressiveness_rating": aggressiveness_rating
    }


    
def classify_team_strength(df, team):
    df = prepare_df(df)

    avg_goals = calculate_team_goal_averages(df)


    sorted_teams = sorted(avg_goals.items(), key=lambda x: x[1], reverse=True)
    total = len(sorted_teams)
    top = set(t for t, _ in sorted_teams[:int(total * 0.3)])
    bottom = set(t for t, _ in sorted_teams[-int(total * 0.3):])

    if team in top:
        return "💪"
    elif team in bottom:
        return "🪶"
    else:
        return "⚖️"
    
def calculate_match_style_score_per_match(df):
    df = prepare_df(df)

    
    # Výpočty
    df["Tempo"] = df["HS"] + df["AS"] + df["HC"] + df["AC"] + df["HF"] + df["AF"]
    df["Goly"] = df["FTHG"] + df["FTAG"]
    df["Konverze"] = (df["FTHG"] + df["FTAG"]) / (df["HST"] + df["AST"]).replace(0, 0.1)
    df["Agrese"] = df["HY"] + df["AY"] + 2 * (df["HR"] + df["AR"]) + df["HF"] + df["AF"]

    # Normalizace
    for col in ["Tempo", "Goly", "Konverze", "Agrese"]:
        col_min = df[col].min()
        col_max = df[col].max()
        df[col + "_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-5)

    # Výpočet skóre
    df["MatchStyleScore"] = (
        0.55 * df["Tempo_norm"] +
        0.35 * df["Goly_norm"] +
        0.10 * df["Konverze_norm"] +
        0.20 * df["Agrese_norm"]
    ) * 100  # škála 0–100

    return df

def expected_match_style_score(df, home_team, away_team, elo_dict, last_n=10, min_required=5):
    """
    Predikuje styl zápasu (MatchStyleScore) na základě medianu z posledních X zápasů
    proti soupeřům s podobným ELO. Používá jen poslední rok, zajišťuje robustnost a loguje počty.
    """
    from datetime import timedelta

    df = calculate_match_style_score_per_match(df)
    df = df.copy()
    latest_date = df['Date'].max()
    one_year_ago = latest_date - timedelta(days=365)
    df = df[df['Date'] >= one_year_ago]

    elo_home = elo_dict.get(home_team, 1500)
    elo_away = elo_dict.get(away_team, 1500)

    def get_similar_matches(team, opponent_elo, team_label):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
        matches['OppELO'] = matches.apply(
            lambda row: elo_dict.get(row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam'], 1500),
            axis=1
        )
        matches['EloDiff'] = abs(matches['OppELO'] - opponent_elo)

        max_elo_diff = 50
        step = 25
        filtered = matches[matches['EloDiff'] <= max_elo_diff]
        while len(filtered) < min_required and max_elo_diff <= 200:
            max_elo_diff += step
            filtered = matches[matches['EloDiff'] <= max_elo_diff]

        if filtered.empty:
            print(f"[{team_label}] ⚠️ Nenalezeny relevantní zápasy — použity všechny dostupné zápasy ({len(matches)})")
            return matches.sort_values("Date", ascending=False).head(last_n)
        else:
            print(f"[{team_label}] ✅ Použito {len(filtered)} zápasů s ELO rozdílem ≤ {max_elo_diff}")
            return filtered.nsmallest(last_n, "EloDiff")

    home_matches = get_similar_matches(home_team, elo_away, "Home team")
    away_matches = get_similar_matches(away_team, elo_home, "Away team")

    home_mss = home_matches["MatchStyleScore"].median() if not home_matches.empty else 0
    away_mss = away_matches["MatchStyleScore"].median() if not away_matches.empty else 0
    expected_mss = round((home_mss + away_mss) / 2, 1)

    # 🎭 Hodnocení
    if expected_mss < 25:
        rating = "🪨 velmi nudné"
    elif expected_mss < 45:
        rating = "😴 nudné"
    elif expected_mss < 65:
        rating = "🎯 středně zajímavé"
    elif expected_mss < 80:
        rating = "⚡ svižné"
    else:
        rating = "🎆 přestřelka"

    return {
        "Expected Match Style Score": expected_mss,
        "Home median": round(home_mss, 1),
        "Away median": round(away_mss, 1),
        "rating": rating
    }



def calculate_gii_zscore(df):
    df = calculate_match_style_score_per_match(df)

    # Výpočet GII komponent
    df["GII_raw"] = (
        df["HS"] + df["AS"] +
        df["HST"] + df["AST"] +
        df["HC"] + df["AC"] +
        df["HF"] + df["AF"] +
        df["HY"] + df["AY"] +
        2 * (df["HR"] + df["AR"])
    )

    # Výpočet Z-skóre v rámci sezóny
    mean = df["GII_raw"].mean()
    std = df["GII_raw"].std()
    df["GII"] = (df["GII_raw"] - mean) / (std + 1e-5)

    return df

def intensity_score_to_emoji(score):
    if score < -1.0:
        return "🪨 extrémně klidné"
    elif score < -0.3:
        return "😴 podprůměrné"
    elif score < 0.3:
        return "🎯 průměr"
    elif score < 1.0:
        return "⚡ svižné"
    else:
        return "🔥 intenzivní show"

def get_team_average_gii(df):
    df = calculate_gii_zscore(df)
    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    team_scores = {}

    for team in teams:
        home_scores = df[df["HomeTeam"] == team]["GII"]
        away_scores = df[df["AwayTeam"] == team]["GII"]
        all_scores = pd.concat([home_scores, away_scores])
        team_scores[team] = round(all_scores.mean(), 2) if not all_scores.empty else 0.0

    return team_scores

def get_goal_probabilities(matrix):
    home_probs = matrix.sum(axis=1)  # řádky = domácí góly
    away_probs = matrix.sum(axis=0)  # sloupce = hostující góly
    return home_probs, away_probs

def detect_risk_factors(df, team, elo_dict):

    warnings = []
    risk_score = 0.0

    # Připravíme data
    df = calculate_match_style_score_per_match(df)

    latest_date = df['Date'].max()

    # Recent form (posledních 5 zápasů)
    last_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)

    points = []
    goals_for = []
    goals_against = []
    shots_on_target = []

    for _, row in last_matches.iterrows():
        if row['HomeTeam'] == team:
            gf, ga, hst = row['FTHG'], row['FTAG'], row['HST']
            is_home = True
        else:
            gf, ga, hst = row['FTAG'], row['FTHG'], row['AST']
            is_home = False

        result = calculate_points(row, is_home)
        points.append(result)
        goals_for.append(gf)
        goals_against.append(ga)
        shots_on_target.append(hst)


    avg_points = np.mean(points) if points else 0
    avg_goals_for = np.mean(goals_for) if goals_for else 0
    avg_goals_against = np.mean(goals_against) if goals_against else 0
    avg_sot = np.mean(shots_on_target) if shots_on_target else 0

    # Sezónní statistiky
    season_matches = df[(df['Date'] >= latest_date - pd.Timedelta(days=365))]
    team_matches = season_matches[(season_matches['HomeTeam'] == team) | (season_matches['AwayTeam'] == team)]

    season_goals_for = []
    season_shots_on_target = []
    season_goals_against = []

    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            season_goals_for.append(row['FTHG'])
            season_goals_against.append(row['FTAG'])
            season_shots_on_target.append(row['HST'])
        else:
            season_goals_for.append(row['FTAG'])
            season_goals_against.append(row['FTHG'])
            season_shots_on_target.append(row['AST'])

    season_avg_goals_for = np.mean(season_goals_for) if season_goals_for else 0
    season_avg_goals_against = np.mean(season_goals_against) if season_goals_against else 0
    season_avg_sot = np.mean(season_shots_on_target) if season_shots_on_target else 0

    # Výpočty poklesů/zhoršení

    ## 1. Form decline
    if avg_points < 1.0:
        warnings.append("❗ Tým je ve špatné formě (méně než 1 bod na zápas).")
        risk_score += 0.25

    ## 2. ELO decline
    current_elo = elo_dict.get(team, 1500)

    one_month_ago = latest_date - pd.Timedelta(days=30)
    df_past = df[df['Date'] <= one_month_ago]
    past_elo_dict = calculate_elo_ratings(df_past)
    past_elo = past_elo_dict.get(team, 1500)

    if (past_elo - current_elo) > 20:
        warnings.append("❗ Tým ztratil více než 20 ELO bodů za poslední měsíc.")
        risk_score += 0.2

    ## 3. xG decline
    if season_avg_sot > 0:
        season_xg = season_avg_sot * 0.1  # jednoduchý xG model
    else:
        season_xg = 0
    if avg_sot > 0:
        recent_xg = avg_sot * 0.1
    else:
        recent_xg = 0

    if season_xg > 0 and recent_xg / season_xg < 0.8:
        warnings.append("❗ Pokles xG o více než 20% oproti sezónnímu průměru.")
        risk_score += 0.2

    ## 4. Finishing problems (góly/střely na branku)
    if avg_sot > 0:
        recent_conversion = avg_goals_for / avg_sot
    else:
        recent_conversion = 0

    if season_avg_sot > 0:
        season_conversion = season_avg_goals_for / season_avg_sot
    else:
        season_conversion = 0

    if season_conversion > 0 and recent_conversion / season_conversion < 0.8:
        warnings.append("❗ Konverze střel se zhoršila o více než 20%.")
        risk_score += 0.2

    ## 5. Defensive collapse (více obdržených gólů)
    if season_avg_goals_against > 0 and avg_goals_against / season_avg_goals_against > 1.2:
        warnings.append("❗ Tým inkasuje o více než 20% více gólů než běžně.")
        risk_score += 0.15

    # Risk score nikdy nesmí být > 1
    risk_score = min(risk_score, 1.0)

    return warnings, risk_score

def calculate_team_styles(df):
    """Vrací dva DataFramy: ofenzivní a defenzivní styl týmů."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

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


def detect_positive_factors(df, team, elo_dict):

    positives = []
    positive_score = 0.0
    df = calculate_match_style_score_per_match(df)
    latest_date = df['Date'].max()

    # Recent matches (last 5)
    last_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)

    points = []
    goals_for = []
    goals_against = []
    shots_on_target = []

    for _, row in last_matches.iterrows():
        if row['HomeTeam'] == team:
            gf, ga, hst = row['FTHG'], row['FTAG'], row['HST']
            is_home = True
        else:
            gf, ga, hst = row['FTAG'], row['FTHG'], row['AST']
            is_home = False

        result = calculate_points(row, is_home)
        points.append(result)
        goals_for.append(gf)
        goals_against.append(ga)
        shots_on_target.append(hst)

    avg_points = np.mean(points) if points else 0
    avg_goals_for = np.mean(goals_for) if goals_for else 0
    avg_goals_against = np.mean(goals_against) if goals_against else 0
    avg_sot = np.mean(shots_on_target) if shots_on_target else 0

    # Season averages
    season_matches = df[(df['Date'] >= latest_date - pd.Timedelta(days=365))]
    team_matches = season_matches[(season_matches['HomeTeam'] == team) | (season_matches['AwayTeam'] == team)]

    season_goals_for = []
    season_shots_on_target = []
    season_goals_against = []

    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            season_goals_for.append(row['FTHG'])
            season_goals_against.append(row['FTAG'])
            season_shots_on_target.append(row['HST'])
        else:
            season_goals_for.append(row['FTAG'])
            season_goals_against.append(row['FTHG'])
            season_shots_on_target.append(row['AST'])

    season_avg_goals_for = np.mean(season_goals_for) if season_goals_for else 0
    season_avg_goals_against = np.mean(season_goals_against) if season_goals_against else 0
    season_avg_sot = np.mean(season_shots_on_target) if season_shots_on_target else 0

    # Výpočty růstů/zlepšení

    ## 1. Form improvement
    if avg_points > 2.0:
        positives.append("🌟 Tým má skvělou formu (více než 2 body na zápas).")
        positive_score += 0.25

    ## 2. ELO improvement
    current_elo = elo_dict.get(team, 1500)

    one_month_ago = latest_date - pd.Timedelta(days=30)
    df_past = df[df['Date'] <= one_month_ago]
    past_elo_dict = calculate_elo_ratings(df_past)
    past_elo = past_elo_dict.get(team, 1500)

    if (current_elo - past_elo) > 20:
        positives.append("🌟 ELO rating týmu vzrostl o více než 20 bodů.")
        positive_score += 0.2

    ## 3. xG growth
    if season_avg_sot > 0:
        season_xg = season_avg_sot * 0.1  # jednoduchý xG model
    else:
        season_xg = 0
    if avg_sot > 0:
        recent_xg = avg_sot * 0.1
    else:
        recent_xg = 0

    if season_xg > 0 and recent_xg / season_xg > 1.2:
        positives.append("🌟 xG týmu vzrostlo o více než 20% oproti sezónnímu průměru.")
        positive_score += 0.2

    ## 4. Finishing improvement
    if avg_sot > 0:
        recent_conversion = avg_goals_for / avg_sot
    else:
        recent_conversion = 0

    if season_avg_sot > 0:
        season_conversion = season_avg_goals_for / season_avg_sot
    else:
        season_conversion = 0

    if season_conversion > 0 and recent_conversion / season_conversion > 1.2:
        positives.append("🌟 Tým výrazně zlepšil konverzi střel.")
        positive_score += 0.2

    ## 5. Defensive improvement
    if season_avg_goals_against > 0 and avg_goals_against / season_avg_goals_against < 0.8:
        positives.append("🌟 Obrana se zlepšila (méně inkasovaných gólů).")
        positive_score += 0.15

    positive_score = min(positive_score, 1.0)

    return positives, positive_score


def calculate_warning_index(df, team, elo_dict):
    warnings = []
    warning_score = 0.0

    df = calculate_match_style_score_per_match(df)

    latest_date = df['Date'].max()

    last_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)
    season_matches = df[(df['Date'] >= latest_date - pd.Timedelta(days=365))]
    season_matches = season_matches[(season_matches['HomeTeam'] == team) | (season_matches['AwayTeam'] == team)]

    def get_stats(matches):
        goals_for = []
        shots_on_target = []
        goals_against = []

        for _, row in matches.iterrows():
            if row['HomeTeam'] == team:
                goals_for.append(row['FTHG'])
                goals_against.append(row['FTAG'])
                shots_on_target.append(row['HST'])
            else:
                goals_for.append(row['FTAG'])
                goals_against.append(row['FTHG'])
                shots_on_target.append(row['AST'])

        return {
            "avg_goals_for": np.mean(goals_for) if goals_for else 0,
            "avg_goals_against": np.mean(goals_against) if goals_against else 0,
            "avg_sot": np.mean(shots_on_target) if shots_on_target else 0
        }

    season_stats = get_stats(season_matches)
    recent_stats = get_stats(last_matches)

    # 1. Pokles xG (proxy přes střely na bránu)
    if season_stats['avg_sot'] > 0 and recent_stats['avg_sot'] / season_stats['avg_sot'] < 0.8:
        warnings.append("Pokles xG >20%")
        warning_score += 0.2

    # 2. Pokles konverze střel
    season_conversion = season_stats['avg_goals_for'] / season_stats['avg_sot'] if season_stats['avg_sot'] > 0 else 0
    recent_conversion = recent_stats['avg_goals_for'] / recent_stats['avg_sot'] if recent_stats['avg_sot'] > 0 else 0

    if season_conversion > 0 and recent_conversion / season_conversion < 0.8:
        warnings.append("Pokles konverze střel >20%")
        warning_score += 0.2

    # 3. Zhoršená obrana
    if season_stats['avg_goals_against'] > 0 and recent_stats['avg_goals_against'] / season_stats['avg_goals_against'] > 1.2:
        warnings.append("Zhoršená obrana (více inkasovaných)")
        warning_score += 0.2

    # 4. ELO pokles
    one_month_ago = latest_date - pd.Timedelta(days=30)
    past_df = df[df['Date'] <= one_month_ago]
    past_elo_dict = calculate_elo_ratings(past_df)
    current_elo = elo_dict.get(team, 1500)
    past_elo = past_elo_dict.get(team, 1500)

    if (past_elo - current_elo) > 20:
        warnings.append("ELO pokles >20 bodů")
        warning_score += 0.2

    # 5. Pokles bodů
    points_last5 = []
    for _, row in last_matches.iterrows():
        if row['HomeTeam'] == team:
            is_home = True
        else:
            is_home = False
        points = calculate_points(row, is_home)
        points_last5.append(points)

    avg_points = np.mean(points_last5) if points_last5 else 0
    if avg_points < 1.0:
        warnings.append("Nízký bodový průměr (<1 bod/zápas)")
        warning_score += 0.2

    warning_score = min(warning_score, 1.0)

    return warnings, warning_score

def calibrate_pseudo_xg_weights(df):
    """
    Vypočítá váhy pro pseudo-xG na základě historických dat v datasetu.
    Vrací tuple: (váha na branku, váha mimo branku)
    """
    df = df.copy()
    df = df[(df['HS'] > 0) | (df['AS'] > 0)]

    home_sot = df['HST'].sum()
    away_sot = df['AST'].sum()
    home_off = (df['HS'] - df['HST']).sum()
    away_off = (df['AS'] - df['AST']).sum()
    total_goals = df['FTHG'].sum() + df['FTAG'].sum()

    total_sot = home_sot + away_sot
    total_off = home_off + away_off

    weight_on_target = total_goals / total_sot if total_sot > 0 else 0.1
    weight_off_target = (total_goals / (total_sot + total_off) * 0.5) if total_off > 0 else 0.05

    return round(weight_on_target, 3), round(weight_off_target, 3)


def detect_overperformance_and_momentum(df, team):
    df = calculate_match_style_score_per_match(df)
    w_on, w_off = calibrate_pseudo_xg_weights(df)

    latest_date = df['Date'].max()
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]

    if team_matches.empty:
        return "N/A", "N/A"

    # Sezónní statistiky
    total_goals = 0
    total_xg = 0
    total_shots = 0
    total_tempo = 0
    total_xg_against = 0
    matches_count = 0

    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            goals = row['FTHG']
            shots = row['HS']
            shots_on_target = row['HST']
            shots_conceded = row['AS']
            shots_on_target_conceded = row['AST']
        else:
            goals = row['FTAG']
            shots = row['AS']
            shots_on_target = row['AST']
            shots_conceded = row['HS']
            shots_on_target_conceded = row['HST']

        # Odvozené hodnoty
        shots_off_target = max(0, shots - shots_on_target)
        shots_conceded_off_target = max(0, shots_conceded - shots_on_target_conceded)

        # Rozšířený xG výpočet s kalibrovanými váhami
        xg = shots_on_target * w_on + shots_off_target * w_off
        xg_conceded = shots_on_target_conceded * w_on + shots_conceded_off_target * w_off

        tempo = shots + shots_conceded + row['HC'] + row['AC'] + row['HF'] + row['AF']

        total_goals += goals
        total_xg += xg
        total_shots += shots
        total_tempo += tempo
        total_xg_against += xg_conceded
        matches_count += 1


    if matches_count == 0:
        return "N/A", "N/A"

    avg_xg = total_xg / matches_count
    avg_shots = total_shots / matches_count
    avg_tempo = total_tempo / matches_count
    avg_xg_against = total_xg_against / matches_count

    # Posledních 5 zápasů
    last5_matches = team_matches.tail(5)

    last5_goals = 0
    last5_xg = 0
    last5_shots = 0
    last5_tempo = 0
    last5_xg_against = 0
    last5_count = 0

    for _, row in last5_matches.iterrows():
        if row['HomeTeam'] == team:
            goals = row['FTHG']
            shots = row['HS']
            shots_on_target = row['HST']
            shots_conceded = row['AS']
            shots_on_target_conceded = row['AST']
        else:
            goals = row['FTAG']
            shots = row['AS']
            shots_on_target = row['AST']
            shots_conceded = row['HS']
            shots_on_target_conceded = row['HST']

        # Počet střel mimo branku (celkem - na branku)
        shots_off_target = shots - shots_on_target
        shots_conceded_off_target = shots_conceded - shots_on_target_conceded

        # Rozšířený pseudo-xG výpočet
        xg = shots_on_target * 0.1 + shots_off_target * 0.05
        xg_conceded = shots_on_target_conceded * 0.1 + shots_conceded_off_target * 0.05

        tempo = shots + shots_conceded + row['HC'] + row['AC'] + row['HF'] + row['AF']

        last5_goals += goals
        last5_xg += xg
        last5_shots += shots
        last5_tempo += tempo
        last5_xg_against += xg_conceded
        last5_count += 1

    if last5_count == 0:
        return "N/A", "N/A"

    avg_last5_xg = last5_xg / last5_count
    avg_last5_shots = last5_shots / last5_count
    avg_last5_tempo = last5_tempo / last5_count
    avg_last5_xg_against = last5_xg_against / last5_count

    # Výpočty
    overperformance = total_goals / total_xg if total_xg > 0 else np.nan

    xg_momentum = avg_last5_xg / avg_xg if avg_xg > 0 else 1
    shots_momentum = avg_last5_shots / avg_shots if avg_shots > 0 else 1
    tempo_momentum = avg_last5_tempo / avg_tempo if avg_tempo > 0 else 1
    defense_momentum = avg_xg_against / avg_last5_xg_against if avg_last5_xg_against > 0 else 1  # pozor, menší xG proti = lepší

    # Vážený průměr momenta
    final_momentum = (
        0.4 * xg_momentum +
        0.3 * shots_momentum +
        0.2 * tempo_momentum +
        0.1 * defense_momentum
    )

    # Interpretace
    if overperformance > 1.2:
        overperf_status = "🚀 Přestřeluje"
    elif overperformance < 0.8:
        overperf_status = "🛑 Podstřeluje"
    else:
        overperf_status = "⚖️ Normální"

    if final_momentum > 1.1:
        momentum_status = "📈 Zlepšuje se"
    elif final_momentum < 0.9:
        momentum_status = "📉 Zhoršuje se"
    else:
        momentum_status = "➖ Stabilní"

    return overperf_status, momentum_status, final_momentum


# Pokud chceš funkci použít víckrát, dej ji např. do helpers.py
def calculate_recent_form_by_matches(df):
    """
    Vrací dictionary: tým -> bodový průměr z posledních 5 zápasů.
    """
    recent_matches = df.copy().sort_values("Date").groupby("HomeTeam").tail(5)
    form_dict = {}
    for team in df["HomeTeam"].unique():
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date").tail(5)
        points = 0
        for _, row in team_matches.iterrows():
            if row["HomeTeam"] == team and row["FTR"] == "H":
                points += 3
            elif row["AwayTeam"] == team and row["FTR"] == "A":
                points += 3
            elif row["FTR"] == "D":
                points += 1
        form_dict[team] = round(points / len(team_matches), 2) if len(team_matches) > 0 else 0
    return form_dict

def scoreline_variance_warning(matrix, threshold=15):
    """
    Vrátí varování, pokud více než `threshold` % pravděpodobnosti připadá na přestřelková skóre typu 2:2, 3:2, 3:3 atd.
    """
    high_scorelines = [(i, j) for i, j in product(range(7), repeat=2) if i + j >= 5 and i > 0 and j > 0]
    total_prob = sum(matrix[i, j] for i, j in high_scorelines)
    if total_prob * 100 > threshold:
        return f"⚠️ {round(total_prob * 100, 1)} % pravděpodobnost přestřelky (např. 2:2, 3:2, 3:3)"
    return None

def combined_form_tempo_warning(df, home_team, away_team, elo_dict, form_dict, tempo_threshold=35, form_threshold=1.5):
    """
    Pokud oba týmy mají vyšší formu a očekávané tempo zápasu je vysoké, upozorní na riziko otevřeného zápasu.
    """
    home_tempo = calculate_match_tempo(df, home_team, elo_dict[away_team], True, elo_dict)["tempo"]
    away_tempo = calculate_match_tempo(df, away_team, elo_dict[home_team], False, elo_dict)["tempo"]
    avg_tempo = (home_tempo + away_tempo) / 2

    home_form = form_dict.get(home_team, 0)
    away_form = form_dict.get(away_team, 0)

    if avg_tempo >= tempo_threshold and home_form > form_threshold and away_form > form_threshold:
        return f"⚠️ Oba týmy mají dobrou formu a zápas má vyšší tempo ({avg_tempo:.1f}) – riziko otevřeného průběhu"
    return None

def conflict_style_warning(df, home_team, away_team, elo_dict):
    """
    Detekuje potenciální konflikt stylů – rychlý a dominantní tým vs trpící soupeř.
    """
    tempo_home = calculate_match_tempo(df, home_team, elo_dict.get(away_team, 1500), True, elo_dict)
    tempo_away = calculate_match_tempo(df, away_team, elo_dict.get(home_team, 1500), False, elo_dict)

    # podmínky pro varování:
    if (
        tempo_home["rating"] in ["Rychlé", "Velmi rychlé"]
        and tempo_home["imbalance_type"] == "Dominantní"
        and tempo_away["imbalance_type"] == "Trpící"
    ):
        return (
            f"⚠️ Stylový konflikt – {home_team} je dominantní s vysokým tempem, "
            f"{away_team} trpící. Riziko rozbití zápasu a přestřelky."
        )
    return None


def get_all_match_warnings(df, home_team, away_team, matrix, elo_dict, form_dict):
    """
    Vrátí seznam všech varování včetně jejich závažnosti (high/medium/low)
    a současně score indexy domácího a hostujícího týmu.
    """
    warnings = []
    score_home = 0
    score_away = 0

    # 1. Rozptyl výsledku (Poisson matice)
    variance_msg = scoreline_variance_warning(matrix)
    if variance_msg:
        warnings.append({"message": variance_msg, "level": "medium"})

    # 2. Forma + tempo konflikt
    style_form_msg = combined_form_tempo_warning(df, home_team, away_team, elo_dict, form_dict)
    if style_form_msg:
        warnings.append({"message": style_form_msg, "level": "medium"})

    # 3. Střet stylů
    style_conflict_msg = conflict_style_warning(df, home_team, away_team, elo_dict)
    if style_conflict_msg:
        warnings.append({"message": style_conflict_msg, "level": "low"})

    # 4. Warning Index – každý tým
    for team in [home_team, away_team]:
        team_warnings, score = calculate_warning_index(df, team, elo_dict)

        if team == home_team:
            score_home = score
        else:
            score_away = score

        if team_warnings:
            severity = "high" if score > 0.6 else "medium" if score > 0.3 else "low"
            warnings.append({
                "message": f"⚠️ {team} Warning Index: {int(score * 100)}% – " + ", ".join(team_warnings),
                "level": severity
            })

    # Seřazení podle závažnosti
    level_priority = {"high": 0, "medium": 1, "low": 2}
    warnings.sort(key=lambda x: level_priority.get(x["level"], 3))

    return warnings, score_home, score_away

def get_all_positive_signals(df, home_team, away_team, elo_dict):
    """
    Vrací seznam pozitivních trendů pro oba týmy (pokud existují).
    """
    from utils.poisson_utils import detect_positive_factors

    positives_summary = []

    for team in [home_team, away_team]:
        positives, _ = detect_positive_factors(df, team, elo_dict)
        if positives:
            positives_summary.append({
                "team": team,
                "messages": positives
            })

    return positives_summary



