
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['FTHG', 'FTAG', 'Date'])
    df = df.sort_values('Date')
    return df

def detect_current_season(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date')
    df = df.dropna(subset=['Date'])

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

# def expected_goals(team_strengths, league_avg_home_goals, league_avg_away_goals, home_team, away_team):
#     home = team_strengths[team_strengths['Team'] == home_team].iloc[0]
#     away = team_strengths[team_strengths['Team'] == away_team].iloc[0]
#     home_exp = league_avg_home_goals * home['AttackHome'] * away['DefenseAway']
#     away_exp = league_avg_away_goals * away['AttackAway'] * home['DefenseHome']
#     return round(home_exp, 2), round(away_exp, 2)

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

def generate_score_heatmap(matrix, home_team, away_team):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(matrix * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={'label': 'Pravděpodobnost (%)'})
    ax.set_title(f"Skóre: {home_team} vs {away_team}")
    ax.set_xlabel(f"{away_team} góly")
    ax.set_ylabel(f"{home_team} góly")
    return fig

def get_top_scorelines(matrix, top_n=5):
    score_probs = [((i, j), matrix[i][j]) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    score_probs.sort(key=lambda x: x[1], reverse=True)
    return score_probs[:top_n]

def plot_top_scorelines(score_probs, home_team, away_team):
    labels = [f"{a}:{b}" for (a, b), _ in score_probs]
    values = [round(p * 100, 2) for _, p in score_probs]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color='skyblue')
    ax.set_title(f"Top skóre: {home_team} vs {away_team}")
    ax.set_xlabel("Skóre")
    ax.set_ylabel("Pravděpodobnost (%)")
    return fig

# def expected_team_stats_combined(df, home_team, away_team, stat_columns):
#     latest_date = df['Date'].max()
#     one_year_ago = latest_date - pd.Timedelta(days=365)
#     season_df = df[df['Date'] >= one_year_ago]

#     output = {}
#     for stat_name, (home_col, away_col) in stat_columns.items():
#         avg_home = df[home_col].mean()
#         avg_away = df[away_col].mean()

#         # Výkonnost za posledních 5 domácích a 5 venkovních zápasů
#         last5_home = df[df['HomeTeam'] == home_team].sort_values('Date').tail(5)
#         last5_away = df[df['AwayTeam'] == away_team].sort_values('Date').tail(5)

#         home_attack = (last5_home[home_col].mean() + season_df[season_df['HomeTeam'] == home_team][home_col].mean()) / 2 / avg_home
#         away_defense = (last5_away[home_col].mean() + season_df[season_df['HomeTeam'] == away_team][home_col].mean()) / 2 / avg_home

#         away_attack = (last5_away[away_col].mean() + season_df[season_df['AwayTeam'] == away_team][away_col].mean()) / 2 / avg_away
#         home_defense = (last5_home[away_col].mean() + season_df[season_df['AwayTeam'] == home_team][away_col].mean()) / 2 / avg_away

#         expected_home = avg_home * home_attack * away_defense
#         expected_away = avg_away * away_attack * home_defense

#         output[stat_name] = {'Home': round(expected_home, 2), 'Away': round(expected_away, 2)}

#     return output





def expected_team_stats_weighted(df, home_team, away_team, stat_columns):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)

    season_df = df[df['Date'] >= one_year_ago]
    historical_df = df[df['Date'] < one_year_ago]

    def safe_mean(df_subset, col):
        if col in df_subset.columns and not df_subset[col].isna().all():
            return df_subset[col].mean()
        return 0.0

    def expected_for_stat(sub_df, home_col, away_col):
        avg_home = safe_mean(sub_df, home_col)
        avg_away = safe_mean(sub_df, away_col)

        if avg_home == 0 or avg_away == 0:
            return 0.0, 0.0

        ha = safe_mean(sub_df[sub_df['HomeTeam'] == home_team], home_col) / avg_home
        hd = safe_mean(sub_df[sub_df['AwayTeam'] == home_team], away_col) / avg_away
        aa = safe_mean(sub_df[sub_df['AwayTeam'] == away_team], away_col) / avg_away
        ad = safe_mean(sub_df[sub_df['HomeTeam'] == away_team], home_col) / avg_home

        expected_home = avg_home * ha * ad
        expected_away = avg_away * aa * hd

        return expected_home, expected_away

    # Posledních 5 zápasů celkem
    last5_home = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].sort_values('Date').tail(5)
    last5_away = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].sort_values('Date').tail(5)
    last5_df = pd.concat([last5_home, last5_away])

    def stat_from_last5(team, col_home, col_away):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date').tail(5)
        vals = []
        for _, row in matches.iterrows():
            if row['HomeTeam'] == team:
                vals.append(row.get(col_home, 0))
            elif row['AwayTeam'] == team:
                vals.append(row.get(col_away, 0))
        return np.mean(vals) if vals else 0.0

    output = {}
    for stat_name, (home_col, away_col) in stat_columns.items():
        if home_col not in df.columns or away_col not in df.columns:
            output[stat_name] = {'Home': 0.0, 'Away': 0.0}
            continue

        hist_home, hist_away = expected_for_stat(historical_df, home_col, away_col)
        season_home, season_away = expected_for_stat(season_df, home_col, away_col)
        last5_home_val = stat_from_last5(home_team, home_col, away_col)
        last5_away_val = stat_from_last5(away_team, home_col, away_col)

        final_home = 0.3 * hist_home + 0.4 * season_home + 0.3 * last5_home_val
        final_away = 0.3 * hist_away + 0.4 * season_away + 0.3 * last5_away_val

        output[stat_name] = {
            'Home': round(final_home, 1),
            'Away': round(final_away, 1)
        }

    return output



def expected_goals_weighted_final(df, home_team, away_team):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)

    season_df = df[df['Date'] >= one_year_ago]
    historical_df = df[df['Date'] < one_year_ago]

    def calculate_team_strengths(df, fallback_df=None):
        all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        strengths = {}

        for team in all_teams:
            home_attack = df[df['HomeTeam'] == team]['FTHG'].mean()
            away_attack = df[df['AwayTeam'] == team]['FTAG'].mean()
            home_defense = df[df['HomeTeam'] == team]['FTAG'].mean()
            away_defense = df[df['AwayTeam'] == team]['FTHG'].mean()

            if all(np.isnan(v) for v in [home_attack, away_attack, home_defense, away_defense]) and fallback_df is not None:
                home_attack = fallback_df[fallback_df['HomeTeam'] == team]['FTHG'].mean()
                away_attack = fallback_df[fallback_df['AwayTeam'] == team]['FTAG'].mean()
                home_defense = fallback_df[fallback_df['HomeTeam'] == team]['FTAG'].mean()
                away_defense = fallback_df[fallback_df['AwayTeam'] == team]['FTHG'].mean()

            strengths[team] = {
                'attack': np.nanmean([home_attack, away_attack]),
                'defense': np.nanmean([home_defense, away_defense])
            }

        return strengths

    def get_expected(df_subset, strengths):
        league_avg_home = df_subset['FTHG'].mean()
        league_avg_away = df_subset['FTAG'].mean()

        if np.isnan(league_avg_home) or np.isnan(league_avg_away):
            return np.nan, np.nan

        ha = np.nanmean([
            df_subset[df_subset['HomeTeam'] == home_team]['FTHG'].mean(),
            df_subset[df_subset['AwayTeam'] == home_team]['FTAG'].mean()
        ]) / league_avg_home

        hd = np.nanmean([
            df_subset[df_subset['AwayTeam'] == home_team]['FTAG'].mean(),
            df_subset[df_subset['HomeTeam'] == home_team]['FTAG'].mean()
        ]) / league_avg_away

        aa = np.nanmean([
            df_subset[df_subset['AwayTeam'] == away_team]['FTAG'].mean(),
            df_subset[df_subset['HomeTeam'] == away_team]['FTHG'].mean()
        ]) / league_avg_away

        ad = np.nanmean([
            df_subset[df_subset['HomeTeam'] == away_team]['FTHG'].mean(),
            df_subset[df_subset['AwayTeam'] == away_team]['FTHG'].mean()
        ]) / league_avg_home

        if any(pd.isna(x) for x in [ha, hd, aa, ad]):
            return np.nan, np.nan

        expected_home = league_avg_home * ha * ad
        expected_away = league_avg_away * aa * hd
        return expected_home, expected_away

    hist_strengths = calculate_team_strengths(historical_df)
    season_strengths = calculate_team_strengths(season_df, fallback_df=df)

    hist_home, hist_away = get_expected(historical_df, hist_strengths)
    season_home, season_away = get_expected(season_df, season_strengths)

    # Posledních 5 zápasů celkem (home i away)
    last5_home = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].sort_values('Date').tail(5)
    last5_away = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].sort_values('Date').tail(5)
    last5_df = pd.concat([last5_home, last5_away])

    # Zde se výpočet obejde bez síly soupeřů
    def expected_from_last5(team, is_home):
        team_matches = last5_df[(last5_df['HomeTeam'] == team) | (last5_df['AwayTeam'] == team)]
        goals_scored = []
        goals_conceded = []

        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                goals_scored.append(row['FTHG'])
                goals_conceded.append(row['FTAG'])
            else:
                goals_scored.append(row['FTAG'])
                goals_conceded.append(row['FTHG'])

        return np.mean(goals_scored), np.mean(goals_conceded)

    last5_home_exp, _ = expected_from_last5(home_team, is_home=True)
    _, last5_away_exp = expected_from_last5(away_team, is_home=False)

    def any_nan(*args):
        return any(pd.isna(x) for x in args)

    if any_nan(hist_home, season_home, last5_home_exp, hist_away, season_away, last5_away_exp):
        raise ValueError(f"❌ Nedostatek dat pro predikci mezi {home_team} a {away_team}.")

    final_home = 0.3 * hist_home + 0.4 * season_home + 0.3 * last5_home_exp
    final_away = 0.3 * hist_away + 0.4 * season_away + 0.3 * last5_away_exp

    return round(final_home, 1), round(final_away, 1)



def validate_dataset(df):
    required_columns = ['Date', 'FTHG', 'FTAG', 'HomeTeam', 'AwayTeam']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Chybí požadované sloupce: {missing}")

    if df.shape[0] < 300:
        st.warning("⚠️ Dataset je příliš malý pro kvalitní predikci (méně než 300 zápasů).")

    if df['Date'].isna().sum() > 0:
        st.warning("⚠️ Některé hodnoty ve sloupci 'Date' nejsou platné a budou odstraněny.")

    if df['FTHG'].isna().sum() > 0 or df['FTAG'].isna().sum() > 0:
        st.warning("⚠️ V některých zápasech chybí výsledky (FTHG nebo FTAG).")

def calculate_team_pseudo_xg(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    results = {}

    # Koeficienty pro různé typy střel (xG model)
    shot_coeffs = {
        "on_target": 0.1,  # Střela na branku
        "off_target": 0.05,  # Střela mimo branku
        "long_distance": 0.05,  # Střela z dálky
        "inside_box": 0.7  # Střela z pokutového území
    }

    for team in teams:
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]
        all_matches = pd.concat([home, away])

        if all_matches.empty:
            results[team] = {
                "avg_xG": 0,
                "xG_home": 0,
                "xG_away": 0,
                "xG_per_goal": 0,
                "xG_total": 0,
                "goals_home": 0,
                "goals_away": 0,
                "conceded_home": 0,
                "conceded_away": 0,
                "xG_total_home": 0,
                "xG_total_away": 0
            }
            continue

        # Počítáme vážený xG pro domácí a venkovní zápasy
        home_xg = (home['HST'] * shot_coeffs["on_target"] + home['HS'] * shot_coeffs["off_target"]).mean()
        away_xg = (away['AST'] * shot_coeffs["on_target"] + away['AS'] * shot_coeffs["off_target"]).mean()

        total_shots = all_matches['HS'].where(all_matches['HomeTeam'] == team, all_matches['AS'])
        total_sot = all_matches['HST'].where(all_matches['HomeTeam'] == team, all_matches['AST'])
        total_goals = all_matches['FTHG'].where(all_matches['HomeTeam'] == team, all_matches['FTAG'])

        # Vypočítáme celkové xG pro tým
        xg_total = (total_sot * shot_coeffs["on_target"] + total_shots * shot_coeffs["off_target"]).sum()
        avg_xg = xg_total / len(all_matches) if len(all_matches) > 0 else 0

        # Počítání gólů a inkasovaných gólů
        total_goals_sum = total_goals.sum()
        goals_home = home['FTHG'].sum()
        goals_away = away['FTAG'].sum()
        conceded_home = home['FTAG'].sum()
        conceded_away = away['FTHG'].sum()

        results[team] = {
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

    return results

    return results


def analyze_opponent_strength(df, team, is_home=True):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    # Výpočet průměru gólů každého týmu (slouží jako "síla")
    team_strength = {}
    for t in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        home_goals = df[df['HomeTeam'] == t]['FTHG'].mean()
        away_goals = df[df['AwayTeam'] == t]['FTAG'].mean()
        avg_goals = np.nanmean([home_goals, away_goals])
        team_strength[t] = avg_goals

    sorted_teams = sorted(team_strength.items(), key=lambda x: x[1], reverse=True)
    total = len(sorted_teams)
    top_30 = set([team for team, _ in sorted_teams[:int(total * 0.3)]])
    bottom_30 = set([team for team, _ in sorted_teams[-int(total * 0.3):]])

    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    opponent_col = 'AwayTeam' if is_home else 'HomeTeam'
    goals_col = 'FTHG' if is_home else 'FTAG'
    shots_col = 'HS' if is_home else 'AS'

    team_matches = df[df[team_col] == team]

    def calc_conversion_against(group):
        group_df = team_matches[team_matches[opponent_col].isin(group)]
        goals = group_df[goals_col].sum()
        shots = group_df[shots_col].sum()
        return round(goals / shots, 2) if shots > 0 else 0

    return {
        'conversion_vs_strong': calc_conversion_against(top_30),
        'conversion_vs_weak': calc_conversion_against(bottom_30),
        'overall_conversion': round(team_matches[goals_col].sum() / team_matches[shots_col].sum(), 2)
        if team_matches[shots_col].sum() > 0 else 0
    }

def calculate_expected_points(outcomes: dict) -> dict:
    """Calculate expected points based on outcome probabilities."""
    home_xp = (outcomes['Home Win'] / 100) * 3 + (outcomes['Draw'] / 100) * 1
    away_xp = (outcomes['Away Win'] / 100) * 3 + (outcomes['Draw'] / 100) * 1
    return {
        'Home xP': round(home_xp, 1),
        'Away xP': round(away_xp, 1)
    }
    
def analyze_opponent_strength(df, team, is_home=True):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    opp_col = 'AwayTeam' if is_home else 'HomeTeam'
    goals_col = 'FTHG' if is_home else 'FTAG'
    shots_col = 'HS' if is_home else 'AS'

    team_matches = df[df[team_col] == team]

    # Výpočet síly všech týmů podle průměru gólů
    avg_goals_per_team = {}
    for t in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        home_goals = df[df['HomeTeam'] == t]['FTHG'].mean()
        away_goals = df[df['AwayTeam'] == t]['FTAG'].mean()
        avg_goals_per_team[t] = np.nanmean([home_goals, away_goals])

    sorted_teams = sorted(avg_goals_per_team.items(), key=lambda x: x[1], reverse=True)
    total = len(sorted_teams)
    top_30 = set([team for team, _ in sorted_teams[:int(total * 0.3)]])
    bottom_30 = set([team for team, _ in sorted_teams[-int(total * 0.3):]])

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

        if opponent in top_30:
            performance['strong'].append(data_point)
        elif opponent in bottom_30:
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
    df = df.copy()
    df = df.sort_values("Date")

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

def calculate_recent_form(df, days=30):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
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
