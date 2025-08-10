import pandas as pd
import numpy as np

from .data import prepare_df
from .stats import calculate_points

def calculate_elo_ratings(df: pd.DataFrame, k: int = 20) -> dict:
    """Spočítá ELO ratingy týmů na základě výsledků zápasů."""
    df = prepare_df(df)
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    elo = {team: 1500 for team in teams}

    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']

        expected_home = 1 / (1 + 10 ** ((elo[away_team] - elo[home_team]) / 400))
        expected_away = 1 / (1 + 10 ** ((elo[home_team] - elo[away_team]) / 400))

        if home_goals > away_goals:
            result_home = 1
            result_away = 0
        elif home_goals < away_goals:
            result_home = 0
            result_away = 1
        else:
            result_home = result_away = 0.5

        elo[home_team] += k * (result_home - expected_home)
        elo[away_team] += k * (result_away - expected_away)

    return elo


def elo_history(df: pd.DataFrame, team: str, k: int = 20) -> pd.DataFrame:
    """Return ELO rating progression for ``team`` after each match.

    The function processes matches chronologically and updates ELO ratings
    after every game. It returns a DataFrame with the ELO value of the given
    team following each match it participated in.

    Args:
        df: DataFrame containing match results.
        team: Team for which the ELO history should be computed.
        k:   K-factor used in the ELO update formula.

    Returns:
        DataFrame with columns ``Date`` and ``ELO`` sorted by date.
    """

    df = prepare_df(df).sort_values("Date")
    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    elo = {t: 1500 for t in teams}
    history = []

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        home_goals = row["FTHG"]
        away_goals = row["FTAG"]

        expected_home = 1 / (1 + 10 ** ((elo[away_team] - elo[home_team]) / 400))
        expected_away = 1 / (1 + 10 ** ((elo[home_team] - elo[away_team]) / 400))

        if home_goals > away_goals:
            result_home, result_away = 1, 0
        elif home_goals < away_goals:
            result_home, result_away = 0, 1
        else:
            result_home = result_away = 0.5

        elo[home_team] += k * (result_home - expected_home)
        elo[away_team] += k * (result_away - expected_away)

        if home_team == team or away_team == team:
            history.append({"Date": row["Date"], "ELO": elo[team]})

    return pd.DataFrame(history)

def calculate_elo_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Vrací DataFrame změn ELO ratingů mezi začátkem a koncem sezóny."""
    df = prepare_df(df)
    start_date = df['Date'].min() + pd.Timedelta(days=5)

    elo_start = calculate_elo_ratings(df[df['Date'] <= start_date])
    elo_end = calculate_elo_ratings(df)

    elo_change = [{"Tým": t, "Změna": round(elo_end.get(t, 1500) - elo_start.get(t, 1500), 1)} for t in elo_end]

    return pd.DataFrame(elo_change).sort_values("Změna", ascending=False).reset_index(drop=True)

def detect_risk_factors(df: pd.DataFrame, team: str, elo_dict: dict) -> tuple:
    """Detekuje rizikové faktory pro daný tým."""
    from .match_style import calculate_match_style_score_per_match

    warnings = []
    risk_score = 0.0

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

    if season_stats['avg_sot'] > 0 and recent_stats['avg_sot'] / season_stats['avg_sot'] < 0.8:
        warnings.append("Pokles xG >20%")
        risk_score += 0.2

    season_conversion = season_stats['avg_goals_for'] / season_stats['avg_sot'] if season_stats['avg_sot'] > 0 else 0
    recent_conversion = recent_stats['avg_goals_for'] / recent_stats['avg_sot'] if recent_stats['avg_sot'] > 0 else 0

    if season_conversion > 0 and recent_conversion / season_conversion < 0.8:
        warnings.append("Pokles konverze střel >20%")
        risk_score += 0.2

    if season_stats['avg_goals_against'] > 0 and recent_stats['avg_goals_against'] / season_stats['avg_goals_against'] > 1.2:
        warnings.append("Zhoršená obrana (více inkasovaných)")
        risk_score += 0.2

    one_month_ago = latest_date - pd.Timedelta(days=30)
    past_df = df[df['Date'] <= one_month_ago]
    past_elo_dict = calculate_elo_ratings(past_df)
    current_elo = elo_dict.get(team, 1500)
    past_elo = past_elo_dict.get(team, 1500)

    if (past_elo - current_elo) > 20:
        warnings.append("ELO pokles >20 bodů")
        risk_score += 0.2

    points_last5 = []
    for _, row in last_matches.iterrows():
        is_home = row['HomeTeam'] == team
        points = calculate_points(row, is_home)
        points_last5.append(points)
    avg_points = np.mean(points_last5) if points_last5 else 0
    if avg_points < 1.0:
        warnings.append("Nízký bodový průměr (<1 bod/zápas)")
        risk_score += 0.2

    risk_score = min(risk_score, 1.0)

    return warnings, risk_score

def detect_positive_factors(df: pd.DataFrame, team: str, elo_dict: dict) -> tuple:
    """Detekuje pozitivní trendy pro daný tým."""
    from .match_style import calculate_match_style_score_per_match

    positive_signals = []
    positive_score = 0.0

    df = calculate_match_style_score_per_match(df)
    latest_date = df['Date'].max()

    last_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)

    points_last5 = []
    goals_for = []
    shots_on_target = []

    for _, row in last_matches.iterrows():
        is_home = row['HomeTeam'] == team
        points = calculate_points(row, is_home)
        points_last5.append(points)

        if is_home:
            goals_for.append(row['FTHG'])
            shots_on_target.append(row['HST'])
        else:
            goals_for.append(row['FTAG'])
            shots_on_target.append(row['AST'])

    avg_points = np.mean(points_last5) if points_last5 else 0
    if avg_points > 2.0:
        positive_signals.append("Silná forma (průměr >2 body)")
        positive_score += 0.4

    avg_conversion = np.mean(goals_for) / np.mean(shots_on_target) if np.mean(shots_on_target) > 0 else 0
    if avg_conversion > 0.4:
        positive_signals.append("Vysoká efektivita střelby")
        positive_score += 0.3

    one_month_ago = latest_date - pd.Timedelta(days=30)
    past_df = df[df['Date'] <= one_month_ago]
    past_elo_dict = calculate_elo_ratings(past_df)
    current_elo = elo_dict.get(team, 1500)
    past_elo = past_elo_dict.get(team, 1500)

    if (current_elo - past_elo) > 20:
        positive_signals.append("Výrazné zlepšení ELO")
        positive_score += 0.3

    positive_score = min(positive_score, 1.0)

    return positive_signals, positive_score

def calculate_warning_index(df: pd.DataFrame, team: str, elo_dict: dict) -> tuple:
    """Spočítá Warning Index (0–1) na základě poklesu xG, konverze, obrany, ELO a bodů."""
    from .data import prepare_df
    from .stats import calculate_points
    from .elo import calculate_elo_ratings
    from .match_style import calculate_match_style_score_per_match

    df = prepare_df(df)
    df = calculate_match_style_score_per_match(df)

    warnings = []
    warning_score = 0.0

    latest_date = df['Date'].max()

    last_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)

    points = []
    goals_for = []
    goals_against = []
    shots_on_target = []

    for _, row in last_matches.iterrows():
        if row['HomeTeam'] == team:
            gf, ga, hst = row['FTHG'], row['FTAG'], row['HST']
            result = 3 if gf > ga else 1 if gf == ga else 0
        else:
            gf, ga, hst = row['FTAG'], row['FTHG'], row['AST']
            result = 3 if gf > ga else 1 if gf == ga else 0
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

    # 1. Pokles střel na branku (proxy na xG)
    if season_avg_sot > 0 and avg_sot / season_avg_sot < 0.8:
        warnings.append("Pokles xG >20%")
        warning_score += 0.2

    # 2. Pokles konverze střel
    season_conversion = season_avg_goals_for / season_avg_sot if season_avg_sot > 0 else 0
    recent_conversion = avg_goals_for / avg_sot if avg_sot > 0 else 0

    if season_conversion > 0 and recent_conversion / season_conversion < 0.8:
        warnings.append("Pokles konverze střel >20%")
        warning_score += 0.2

    # 3. Zhoršená obrana
    if season_avg_goals_against > 0 and avg_goals_against / season_avg_goals_against > 1.2:
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

    # 5. Pokles bodového zisku
    if avg_points < 1.0:
        warnings.append("Nízký bodový průměr (<1 bod/zápas)")
        warning_score += 0.2

    warning_score = min(warning_score, 1.0)

    return warnings, warning_score

def detect_overperformance_and_momentum(df: pd.DataFrame, team: str) -> tuple:
    """Detekuje, zda tým aktuálně overperformuje a jeho momentum (s emoji a CZ výstupem)."""
    from .data import prepare_df
    from .stats import calculate_points
    import numpy as np

    df = prepare_df(df)

    last_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(10)

    xG_diff = []
    points_list = []

    for _, row in last_matches.iterrows():
        if row['HomeTeam'] == team:
            goals = row['FTHG']
            shots = row['HS']
            shots_on_target = row['HST']
        else:
            goals = row['FTAG']
            shots = row['AS']
            shots_on_target = row['AST']

        conv = goals / shots_on_target if shots_on_target > 0 else 0
        pseudo_xg = shots_on_target * conv
        diff = goals - pseudo_xg
        xG_diff.append(diff)

        is_home = row['HomeTeam'] == team
        points = calculate_points(row, is_home)
        points_list.append(points)

    avg_xg_diff = np.mean(xG_diff) if xG_diff else 0
    avg_points = np.mean(points_list) if points_list else 0

    # 🎯 Overperformance
    if avg_xg_diff > 0.5:
        overperformance = "🚀"
    elif avg_xg_diff < -0.5:
        overperformance = "🐌"
    else:
        overperformance = "⚖️"

    # 🔄 Momentum
    if avg_points > 2.0:
        momentum = "📈"
    elif avg_points < 1.0:
        momentum = "📉"
    else:
        momentum = "⚪"

    return overperformance, momentum

