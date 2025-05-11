import pandas as pd
import numpy as np
import math
from scipy.stats import poisson


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Základní úprava dat: kopírování, převod datumu, odstranění nevalidních řádků, seřazení podle data."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    #df = df.dropna(subset=['Date'])
    # Odstraň řádky bez datumu nebo týmů
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]
    df = df[df["AwayTeam"].astype(str).str.strip() != ""]
    df = df.sort_values('Date')
    return df

def load_data(file_path: str) -> pd.DataFrame:
    """Načte CSV soubor a připraví ho."""
    df = pd.read_csv(file_path)
    df = prepare_df(df)
    required_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df

# def aggregate_team_stats(df: pd.DataFrame) -> pd.DataFrame:
#     """Vrací agregované týmové statistiky podle domácích zápasů."""
#     df = df.copy()
#     team_stats = df.groupby("HomeTeam").agg({
#         "FTHG": "mean",
#         "FTAG": "mean",
#         "HS": "mean",
#         "HST": "mean",
#         "HC": "mean",
#         "HY": "mean"
#     }).rename(columns={
#         "FTHG": "Góly doma",
#         "FTAG": "Góly venku",
#         "HS": "Střely",
#         "HST": "Na branku",
#         "HC": "Rohy",
#         "HY": "Žluté"
#     })
#     return team_stats

def aggregate_team_stats(df):
    """Agreguje statistiky za všechny zápasy (doma i venku) pro každý tým."""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    records = []

    for team in teams:
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]

        goals = pd.concat([home['FTHG'], away['FTAG']])
        conceded = pd.concat([home['FTAG'], away['FTHG']])
        shots = pd.concat([home['HS'], away['AS']])
        shots_on_target = pd.concat([home['HST'], away['AST']])
        corners = pd.concat([home['HC'], away['AC']])
        yellows = pd.concat([home['HY'], away['AY']])

        records.append({
            "Tým": team,
            "Góly": goals.mean(),
            "Obdržené góly": conceded.mean(),
            "Střely": shots.mean(),
            "Na branku": shots_on_target.mean(),
            "Rohy": corners.mean(),
            "Žluté": yellows.mean()
        })

    df_stats = pd.DataFrame(records).set_index("Tým")

    print(df_stats.columns)
    return df_stats


def detect_current_season(df: pd.DataFrame) -> tuple:
    """Detekuje aktuální sezónu podle nejnovějšího data."""
    df = prepare_df(df)
    latest_date = df['Date'].max()
    if latest_date.month > 6:
        season_start = pd.Timestamp(year=latest_date.year, month=8, day=1)
    else:
        season_start = pd.Timestamp(year=latest_date.year - 1, month=8, day=1)
    season_df = df[df['Date'] >= season_start]
    return season_df, season_start

def calculate_points(row: pd.Series, is_home: bool) -> int:
    """Spočítá body za zápas."""
    if is_home:
        if row['FTHG'] > row['FTAG']:
            return 3
        elif row['FTHG'] == row['FTAG']:
            return 1
        else:
            return 0
    else:
        if row['FTAG'] > row['FTHG']:
            return 3
        elif row['FTAG'] == row['FTHG']:
            return 1
        else:
            return 0

def add_btts_column(df: pd.DataFrame) -> pd.DataFrame:
    """Přidá sloupec 'BTTS' indikující, zda oba týmy skórovaly."""
    df = df.copy()
    df['BTTS'] = df.apply(lambda row: int(row['FTHG'] > 0 and row['FTAG'] > 0), axis=1)
    return df

def poisson_prediction(home_exp_goals: float, away_exp_goals: float, max_goals: int = 6) -> np.ndarray:
    """Vrací Poissonovu pravděpodobnost výsledků do maximálního počtu gólů."""
    home_goals_probs = [poisson_pmf(home_exp_goals, i) for i in range(max_goals + 1)]
    away_goals_probs = [poisson_pmf(away_exp_goals, i) for i in range(max_goals + 1)]
    matrix = np.outer(home_goals_probs, away_goals_probs)
    return matrix

def poisson_pmf(lmbda: float, k: int) -> float:
    """Poissonova pravděpodobnostní funkce."""
    return (lmbda ** k) * np.exp(-lmbda) / math.factorial(k)

def prob_to_odds(prob: float) -> str:
    """Převede pravděpodobnost (v procentech) na desetinný kurz."""
    if prob <= 0:
        return "-"
    decimal_odds = 100 / prob
    return f"{decimal_odds:.2f}"


def calculate_expected_points(outcomes: dict) -> dict:
    """Calculate expected points based on outcome probabilities."""
    home_xp = (outcomes['Home Win'] / 100) * 3 + (outcomes['Draw'] / 100) * 1
    away_xp = (outcomes['Away Win'] / 100) * 3 + (outcomes['Draw'] / 100) * 1
    return {
        'Home xP': round(home_xp, 1),
        'Away xP': round(away_xp, 1)
    }

def calculate_team_strengths(df: pd.DataFrame) -> tuple:
    """Spočítá útočnou a obrannou sílu týmů na základě gólů."""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    attack_strength = {}
    defense_strength = {}
    for team in teams:
        home_scored = df[df['HomeTeam'] == team]['FTHG'].mean()
        away_scored = df[df['AwayTeam'] == team]['FTAG'].mean()
        home_conceded = df[df['HomeTeam'] == team]['FTAG'].mean()
        away_conceded = df[df['AwayTeam'] == team]['FTHG'].mean()

        attack_strength[team] = np.nanmean([home_scored, away_scored])
        defense_strength[team] = np.nanmean([home_conceded, away_conceded])

    league_attack_avg = np.nanmean(list(attack_strength.values()))
    league_defense_avg = np.nanmean(list(defense_strength.values()))

    return attack_strength, defense_strength, (league_attack_avg, league_defense_avg)

def classify_team_strength(df: pd.DataFrame, team: str) -> str:
    """Klasifikuje tým podle průměrného počtu gólů (silný, průměrný, slabý)."""
    avg_goals = {}
    for t in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        home_avg = df[df['HomeTeam'] == t]['FTHG'].mean()
        away_avg = df[df['AwayTeam'] == t]['FTAG'].mean()
        avg_goals[t] = np.nanmean([home_avg, away_avg])

    sorted_teams = sorted(avg_goals.items(), key=lambda x: x[1], reverse=True)
    total = len(sorted_teams)
    top_30 = set([t for t, _ in sorted_teams[:int(total * 0.3)]])
    bottom_30 = set([t for t, _ in sorted_teams[-int(total * 0.3):]])

    if team in top_30:
        return "Silní"
    elif team in bottom_30:
        return "Slabí"
    else:
        return "Průměrní"
    

def compute_form_trend(score_list):
    """
    Vrací emoji podle vývoje formy (rozdíl bodů mezi posledními 3 a předchozími 6 zápasy).
    """
    if len(score_list) < 9:
        return "❓"

    recent = score_list[-3:]        # poslední 3 zápasy
    earlier = score_list[-9:-3]     # předchozích 6 zápasů

    def calc_points(results):
        return sum([3 if gf > ga else 1 if gf == ga else 0 for gf, ga in results])

    recent_points = calc_points(recent)
    earlier_points = calc_points(earlier)

    avg_recent = recent_points / 3
    avg_earlier = earlier_points / 6

    delta = avg_recent - avg_earlier

    if delta >= 1:
        return "📈"
    elif delta <= -1:
        return "📉"
    else:
        return "➖"



def compute_score_stats(df: pd.DataFrame, team: str):
    """
    Vrací tuple: (list výsledků), průměr gólů na zápas, rozptyl skóre
    """
    team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date").tail(10)

    score_list = []
    total_scored = 0
    total_conceded = 0

    for _, row in team_matches.iterrows():
        if row["HomeTeam"] == team:
            gf = row["FTHG"]
            ga = row["FTAG"]
        else:
            gf = row["FTAG"]
            ga = row["FTHG"]

        score_list.append((gf, ga))
        total_scored += gf
        total_conceded += ga

    avg_goals_per_match = (total_scored + total_conceded) / len(score_list) if score_list else 0
    score_variance = np.var([gf + ga for gf, ga in score_list]) if score_list else 0

    return score_list, avg_goals_per_match, score_variance

def poisson_over25_probability(home_exp, away_exp):
    matrix = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            matrix[i][j] = poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)

    prob_over = sum(matrix[i][j] for i in range(7) for j in range(7) if i + j > 2.5)
    return round(prob_over * 100, 2)

def expected_goals_vs_similar_elo_weighted(df, home_team, away_team, elo_dict, elo_tolerance=50):
    df = prepare_df(df)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])

    elo_home = elo_dict.get(home_team, 1500)
    elo_away = elo_dict.get(away_team, 1500)

    today = df['Date'].max()
    df['HomeELO'] = df['HomeTeam'].map(elo_dict)
    df['AwayELO'] = df['AwayTeam'].map(elo_dict)
    df['days_ago'] = (today - df['Date']).dt.days
    df['weight'] = 1 / (df['days_ago'] + 1)

    # --- Filtrujeme relevantní zápasy pro každý případ ---
    df_home_relevant = df[(df['HomeTeam'] == home_team) & (abs(df['AwayELO'] - elo_away) <= elo_tolerance)].copy()
    df_away_relevant = df[(df['AwayTeam'] == away_team) & (abs(df['HomeELO'] - elo_home) <= elo_tolerance)].copy()

    df_home_all = df[((df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)) & 
                     ((abs(df['AwayELO'] - elo_away) <= elo_tolerance) | (abs(df['HomeELO'] - elo_away) <= elo_tolerance))].copy()

    df_away_all = df[((df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)) & 
                     ((abs(df['HomeELO'] - elo_home) <= elo_tolerance) | (abs(df['AwayELO'] - elo_home) <= elo_tolerance))].copy()

    league_avg_home = df['FTHG'].mean()
    league_avg_away = df['FTAG'].mean()

    def weighted_stat(goals, weights):
        return np.average(goals, weights=weights) if len(goals) > 0 else 1.0

    # --- Výpočet home only ---
    gf_home = weighted_stat(df_home_relevant['FTHG'], df_home_relevant['weight'])
    ga_home = weighted_stat(df_home_relevant['FTAG'], df_home_relevant['weight'])

    # --- Výpočet away only ---
    gf_away = weighted_stat(df_away_relevant['FTAG'], df_away_relevant['weight'])
    ga_away = weighted_stat(df_away_relevant['FTHG'], df_away_relevant['weight'])

    # --- Výpočet all matches home ---
    gf_home_all = weighted_stat(
        df_home_all.apply(lambda row: row['FTHG'] if row['HomeTeam'] == home_team else row['FTAG'], axis=1),
        df_home_all['weight']
    )
    ga_home_all = weighted_stat(
        df_home_all.apply(lambda row: row['FTAG'] if row['HomeTeam'] == home_team else row['FTHG'], axis=1),
        df_home_all['weight']
    )

    # --- Výpočet all matches away ---
    gf_away_all = weighted_stat(
        df_away_all.apply(lambda row: row['FTAG'] if row['AwayTeam'] == away_team else row['FTHG'], axis=1),
        df_away_all['weight']
    )
    ga_away_all = weighted_stat(
        df_away_all.apply(lambda row: row['FTHG'] if row['AwayTeam'] == away_team else row['FTAG'], axis=1),
        df_away_all['weight']
    )

    def compute_expected(gf, ga_opp, l_home, l_away):
        return l_home * (gf / l_home) * (ga_opp / l_away)

    # Výpočty
    home_exp_home = compute_expected(gf_home, ga_away, league_avg_home, league_avg_away)
    away_exp_away = compute_expected(gf_away, ga_home, league_avg_away, league_avg_home)

    home_exp_all = compute_expected(gf_home_all, ga_away_all, league_avg_home, league_avg_away)
    away_exp_all = compute_expected(gf_away_all, ga_home_all, league_avg_away, league_avg_home)

    # Výpis
    print("📘 ELO-based: Home/Away only")
    print(f"  HomeExp: {home_exp_home:.2f}, AwayExp: {away_exp_away:.2f} → Over 2.5: {poisson_over25_probability(home_exp_home, away_exp_away)}%")

    print("📘 ELO-based: All relevant matches")
    print(f"  HomeExp: {home_exp_all:.2f}, AwayExp: {away_exp_all:.2f} → Over 2.5: {poisson_over25_probability(home_exp_all, away_exp_all)}%")

    # Pro kombinovaný výstup vracíme průměr obou přístupů
    combined_home = round((home_exp_home + home_exp_all) / 2, 2)
    combined_away = round((away_exp_away + away_exp_all) / 2, 2)

    print("🎯 ELO-based kombinace")
    print(f"  FinalExp: {combined_home:.2f} - {combined_away:.2f} → Over 2.5: {poisson_over25_probability(combined_home, combined_away)}%")

    return combined_home, combined_away

def get_last_n_matches(df, team, role="both", n=10):
    if role == "home":
        matches = df[df['HomeTeam'] == team]
    elif role == "away":
        matches = df[df['AwayTeam'] == team]
    else:
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    return matches.sort_values("Date").tail(n)

