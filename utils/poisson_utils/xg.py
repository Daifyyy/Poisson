import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from .data import prepare_df
from .prediction import poisson_prediction


# Caches for expected goals and filtered match datasets
_expected_goals_cache: Dict[str, Dict[Tuple[str, str], Tuple[float, float]]] = {}
_filtered_matches_cache: Dict[str, Dict[Tuple[str, str, bool], pd.DataFrame]] = {}
_league_versions: Dict[str, Tuple[int, pd.Timestamp]] = {}

def calculate_pseudo_xg_for_team(df: pd.DataFrame, team: str) -> dict:
    """Počítá pseudo-xG metriky pro jeden tým."""
    df = prepare_df(df)

    home_matches = df[df['HomeTeam'] == team]
    away_matches = df[df['AwayTeam'] == team]

    shot_coeffs = {
        "on_target": 0.1,  # střely na branku
        "off_target": 0.05  # ostatní střely
    }

    home_xg = (home_matches['HST'] * shot_coeffs["on_target"] + home_matches['HS'] * shot_coeffs["off_target"]).mean()
    away_xg = (away_matches['AST'] * shot_coeffs["on_target"] + away_matches['AS'] * shot_coeffs["off_target"]).mean()

    total_shots = df['HS'].where(df['HomeTeam'] == team, df['AS'])
    total_sot = df['HST'].where(df['HomeTeam'] == team, df['AST'])
    total_goals = df['FTHG'].where(df['HomeTeam'] == team, df['FTAG'])

    xg_total = (total_sot * shot_coeffs["on_target"] + total_shots * shot_coeffs["off_target"]).sum()
    avg_xg = xg_total / len(df) if len(df) > 0 else 0

    total_goals_sum = total_goals.sum()
    goals_home = home_matches['FTHG'].sum()
    goals_away = away_matches['FTAG'].sum()
    conceded_home = home_matches['FTAG'].sum()
    conceded_away = away_matches['FTHG'].sum()

    return {
        "avg_xG": round(avg_xg, 1),
        "xG_home": round(home_xg, 1),
        "xG_away": round(away_xg, 1),
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
    """
    Hrubý výpočet pseudo xG a xGA na základě střel a střel na bránu.
    xG = 0.1 * střely + 0.3 * střely na branku
    xGA = stejný výpočet na základě střel soupeře
    """

    # vytvoření společného DataFrame pro domácí i venkovní týmy
    home = df[['HomeTeam', 'HS', 'HST', 'AS', 'AST']].rename(
        columns={
            'HomeTeam': 'Team',
            'HS': 'shots_for',
            'HST': 'shots_on_target_for',
            'AS': 'shots_against',
            'AST': 'shots_on_target_against',
        }
    )
    away = df[['AwayTeam', 'AS', 'AST', 'HS', 'HST']].rename(
        columns={
            'AwayTeam': 'Team',
            'AS': 'shots_for',
            'AST': 'shots_on_target_for',
            'HS': 'shots_against',
            'HST': 'shots_on_target_against',
        }
    )
    combined = pd.concat([home, away], ignore_index=True)

    # agregace statistik pro každý tým v jednom průchodu
    totals = combined.groupby('Team').agg(
        shots_for=('shots_for', 'sum'),
        shots_on_target_for=('shots_on_target_for', 'sum'),
        shots_against=('shots_against', 'sum'),
        shots_on_target_against=('shots_on_target_against', 'sum'),
        matches=('shots_for', 'size'),
    )

    # výpočet pseudo xG a xGA
    totals['xg'] = 0.1 * totals['shots_for'] + 0.3 * totals['shots_on_target_for']
    totals['xga'] = 0.1 * totals['shots_against'] + 0.3 * totals['shots_on_target_against']

    # přepočet na průměr na zápas a převod na slovník
    totals['xg'] = (totals['xg'] / totals['matches']).round(2)
    totals['xga'] = (totals['xga'] / totals['matches']).round(2)

    return totals[['xg', 'xga']].to_dict('index')


def expected_goals_weighted_by_elo(df: pd.DataFrame, home_team: str, away_team: str, elo_dict: dict) -> tuple:
    """Calculate expected goals for both teams using ELO-weighted averages.

    Results are memoized per ``(home_team, away_team, league)`` combination.
    Filtered datasets for historical and seasonal samples are cached so repeated
    calls avoid re-sorting large dataframes.  Cache is invalidated whenever the
    underlying league dataframe changes (length or max date).
    """

    df = prepare_df(df)

    league = df['Div'].iloc[0] if 'Div' in df.columns else 'unknown'
    version = (len(df), df['Date'].max())

    if _league_versions.get(league) != version:
        # League data changed -> invalidate caches for this league
        _expected_goals_cache.pop(league, None)
        _filtered_matches_cache.pop(league, None)
        _league_versions[league] = version

    league_cache = _expected_goals_cache.setdefault(league, {})
    cache_key = (home_team, away_team)
    if cache_key in league_cache:
        return league_cache[cache_key]

    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)
    df_hist = df[df['Date'] < one_year_ago]
    df_season = df[df['Date'] >= one_year_ago]
    df_last10 = df[
        (df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team) |
        (df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)
    ].sort_values('Date').tail(10)

    filtered_cache = _filtered_matches_cache.setdefault(league, {})

    def get_matches(sub_df: pd.DataFrame, subset_key: Optional[str], team: str, is_home: bool) -> pd.DataFrame:
        team_col = 'HomeTeam' if is_home else 'AwayTeam'
        opp_col = 'AwayTeam' if is_home else 'HomeTeam'

        if subset_key is None:
            matches = sub_df[sub_df[team_col] == team].copy()
            matches['OppELO'] = matches[opp_col].map(elo_dict)
            return matches

        key = (subset_key, team, is_home)
        if key not in filtered_cache:
            matches = sub_df[sub_df[team_col] == team].copy()
            matches['OppELO'] = matches[opp_col].map(elo_dict)
            filtered_cache[key] = matches
        return filtered_cache[key]

    def compute_stats(sub_df: pd.DataFrame, subset_key: Optional[str], team: str, is_home: bool, opponent_elo: float, n: int = 10):
        gf_col = 'FTHG' if is_home else 'FTAG'
        ga_col = 'FTAG' if is_home else 'FTHG'

        matches = get_matches(sub_df, subset_key, team, is_home)
        if matches.empty:
            return 1.0, 1.0

        tmp = matches.assign(EloDiff=(matches['OppELO'] - opponent_elo).abs())
        closest = tmp.nsmallest(n, 'EloDiff')

        gf = closest[gf_col].mean() if not closest.empty else 1.0
        ga = closest[ga_col].mean() if not closest.empty else 1.0
        return gf, ga

    elo_away = elo_dict.get(away_team, 1500)
    elo_home = elo_dict.get(home_team, 1500)

    hist_home, hist_home_ga = compute_stats(df_hist, 'hist', home_team, True, elo_away)
    hist_away, hist_away_ga = compute_stats(df_hist, 'hist', away_team, False, elo_home)

    season_home, season_home_ga = compute_stats(df_season, 'season', home_team, True, elo_away)
    season_away, season_away_ga = compute_stats(df_season, 'season', away_team, False, elo_home)

    last5_home, last5_home_ga = compute_stats(df_last10, None, home_team, True, elo_away)
    last5_away, last5_away_ga = compute_stats(df_last10, None, away_team, False, elo_home)

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

    result = round(expected_home, 2), round(expected_away, 2)
    league_cache[cache_key] = result
    return result


def expected_team_stats_weighted_by_elo(df: pd.DataFrame, home_team: str, away_team: str, stat_columns: dict, elo_dict: dict) -> dict:
    """Vrací očekávané týmové statistiky na základě 10 nejbližších soupeřů podle ELO rozdílu."""
    df = prepare_df(df)

    # Získání ELO soupeřů
    elo_away = elo_dict.get(away_team, 1500)
    elo_home = elo_dict.get(home_team, 1500)

    def filter_matches(team: str, is_home: bool, opponent_elo: float) -> pd.DataFrame:
        team_col = 'HomeTeam' if is_home else 'AwayTeam'
        opp_col = 'AwayTeam' if is_home else 'HomeTeam'
        matches = df[df[team_col] == team].copy()
        matches['OppELO'] = matches[opp_col].map(elo_dict)
        matches['EloDiff'] = abs(matches['OppELO'] - opponent_elo)
        return matches.sort_values('EloDiff').head(10)

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



def poisson_prediction_matrix(home_xg: float, away_xg: float, max_goals: int = 6) -> np.ndarray:
    """Vrací Poissonovu predikční matici výsledků."""
    return poisson_prediction(home_xg, away_xg, max_goals)

def bt_btts_prob(df: pd.DataFrame, home_team: str, away_team: str, elo_dict: dict) -> dict:
    """Spočítá pravděpodobnosti BTTS (Both Teams To Score)."""
    home_xg, away_xg = expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
    matrix = poisson_prediction_matrix(home_xg, away_xg)

    no_goal = matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0]
    btts_prob = 1 - no_goal

    return {"btts_yes": btts_prob, "btts_no": 1 - btts_prob}

def match_outcomes_prob(df: pd.DataFrame, home_team: str, away_team: str, elo_dict: dict) -> dict:
    """Spočítá pravděpodobnosti výsledků (domácí výhra, remíza, hosté)."""
    home_xg, away_xg = expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
    matrix = poisson_prediction_matrix(home_xg, away_xg)

    home_win = np.tril(matrix, k=-1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, k=1).sum()

    return {"home_win": home_win, "draw": draw, "away_win": away_win}

def over_under_prob(matrix: np.ndarray, threshold: float) -> dict:
    """Compute probabilities for an Over/Under goal line.

    Parameters
    ----------
    matrix : np.ndarray
        Poisson probability matrix for scorelines.
    threshold : float
        Goal threshold (e.g. 2.5 for the classic line).

    Returns
    -------
    dict
        Percentage probabilities for going over or under the threshold.
    """

    goals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    over = float(matrix[goals > threshold].sum())
    under = float(matrix[goals <= threshold].sum())

    return {
        f"Over {threshold}": round(over * 100, 1),
        f"Under {threshold}": round(under * 100, 1),
    }


def get_goal_probabilities(df: pd.DataFrame, home_team: str, away_team: str, elo_dict: dict) -> np.ndarray:
    """Vrací celou predikční Poisson matici pro zápas."""
    home_xg, away_xg = expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
    return poisson_prediction_matrix(home_xg, away_xg)

def get_goal_probabilities(matrix: np.ndarray) -> tuple:
    """Vrací pravděpodobnosti pro domácí a hostující tým, že dají N gólů."""
    home_probs = matrix.sum(axis=1)  # Součet přes sloupce = góly domácích
    away_probs = matrix.sum(axis=0)  # Součet přes řádky = góly hostujících

    return home_probs, away_probs


def match_outcomes_prob(matrix: np.ndarray) -> dict:
    """Vrací pravděpodobnosti výhry domácích, remízy a výhry hostů z Poisson matice."""
    home_win = np.tril(matrix, k=-1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, k=1).sum()

    return {
        "Home Win": round(home_win * 100, 1),
        "Draw": round(draw * 100, 1),
        "Away Win": round(away_win * 100, 1)
    }


def get_top_scorelines(matrix: np.ndarray, top_n: int = 5) -> list:
    """Vrací top N nejpravděpodobnějších výsledků ze skore matrixe."""
    scorelines = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            scorelines.append(((i, j), matrix[i, j]))
    scorelines.sort(key=lambda x: x[1], reverse=True)
    return scorelines[:top_n]

def btts_prob(matrix: np.ndarray) -> dict:
    """Spočítá pravděpodobnosti BTTS přímo z matrixe výsledků."""
    no_goal = matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0]
    btts = 1 - no_goal
    return {"BTTS Yes": round(btts * 100, 1), "BTTS No": round((1 - btts) * 100, 1)}


def additional_opportunities_prob(matrix: np.ndarray) -> dict:
    """Pravděpodobnosti pro vybrané kombinované sázky.

    Vrací procentuální pravděpodobnosti pro následující příležitosti:

    - výhra domácích a více než 1.5 gólu
    - výhra hostů a více než 1.5 gólu
    - výhra domácích v handicapu -1.0 (alespoň o 2 góly)
    - výhra hostů v handicapu -1.0 (alespoň o 2 góly)
    """

    goals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    home_goals = np.arange(matrix.shape[0])[:, None]
    away_goals = np.arange(matrix.shape[1])[None, :]

    home_over15 = matrix[(home_goals > away_goals) & (goals >= 2)].sum()
    away_over15 = matrix[(away_goals > home_goals) & (goals >= 2)].sum()
    home_minus1 = matrix[(home_goals >= away_goals + 2)].sum()
    away_minus1 = matrix[(away_goals >= home_goals + 2)].sum()

    return {
        "Home Win & Over 1.5": round(home_over15 * 100, 1),
        "Away Win & Over 1.5": round(away_over15 * 100, 1),
        "Home -1.0": round(home_minus1 * 100, 1),
        "Away -1.0": round(away_minus1 * 100, 1),
    }
