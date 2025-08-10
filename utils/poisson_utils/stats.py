import pandas as pd
import numpy as np


def aggregate_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Agreguje statistiky za všechny zápasy (doma i venku) pro každý tým."""
    df = df.copy()
    # inicializace chybějících sloupců, aby je bylo možné agregovat
    for col in ["HY", "AY", "HR", "AR", "HF", "AF"]:
        if col not in df.columns:
            df[col] = 0

    # připravení dat v dlouhém formátu pro přímou agregaci
    home_cols = {
        "HomeTeam": "Tým",
        "FTHG": "Góly",
        "FTAG": "Obdržené góly",
        "HS": "Střely",
        "HST": "Na branku",
        "HC": "Rohy",
        "HY": "Žluté",
        "HR": "Červené",
        "HF": "Fauly",
    }
    away_cols = {
        "AwayTeam": "Tým",
        "FTAG": "Góly",
        "FTHG": "Obdržené góly",
        "AS": "Střely",
        "AST": "Na branku",
        "AC": "Rohy",
        "AY": "Žluté",
        "AR": "Červené",
        "AF": "Fauly",
    }

    home_df = df[list(home_cols.keys())].rename(columns=home_cols)
    away_df = df[list(away_cols.keys())].rename(columns=away_cols)
    stats = pd.concat([home_df, away_df], ignore_index=True)

    return stats.groupby("Tým").mean()


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
    df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    return df


def calculate_team_strengths(df: pd.DataFrame) -> tuple:
    """Spočítá útočnou a obrannou sílu týmů na základě gólů."""
    home_stats = (
        df.groupby("HomeTeam")[["FTHG", "FTAG"]].mean()
        .rename(columns={"FTHG": "scored_home", "FTAG": "conceded_home"})
    )
    away_stats = (
        df.groupby("AwayTeam")[["FTAG", "FTHG"]].mean()
        .rename(columns={"FTAG": "scored_away", "FTHG": "conceded_away"})
    )
    stats = home_stats.join(away_stats, how="outer")
    stats["attack"] = stats[["scored_home", "scored_away"]].mean(axis=1)
    stats["defense"] = stats[["conceded_home", "conceded_away"]].mean(axis=1)
    league_attack_avg = stats["attack"].mean()
    league_defense_avg = stats["defense"].mean()
    attack_strength = stats["attack"].to_dict()
    defense_strength = stats["defense"].to_dict()
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
    """Vrací emoji podle vývoje formy (rozdíl bodů mezi posledními 3 a předchozími 6 zápasy)."""
    if len(score_list) < 9:
        return "❓"

    recent = score_list[-3:]
    earlier = score_list[-9:-3]

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
    """Vrací tuple: (list výsledků), průměr gólů na zápas, rozptyl skóre"""
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
