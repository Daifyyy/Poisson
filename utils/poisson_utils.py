
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

def expected_goals(team_strengths, league_avg_home_goals, league_avg_away_goals, home_team, away_team):
    home = team_strengths[team_strengths['Team'] == home_team].iloc[0]
    away = team_strengths[team_strengths['Team'] == away_team].iloc[0]
    home_exp = league_avg_home_goals * home['AttackHome'] * away['DefenseAway']
    away_exp = league_avg_away_goals * away['AttackAway'] * home['DefenseHome']
    return round(home_exp, 2), round(away_exp, 2)

def poisson_prediction(home_exp, away_exp, max_goals=5):
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

def expected_team_stats_combined(df, home_team, away_team, stat_columns):
    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)
    season_df = df[df['Date'] >= one_year_ago]

    output = {}
    for stat_name, (home_col, away_col) in stat_columns.items():
        avg_home = df[home_col].mean()
        avg_away = df[away_col].mean()

        # Výkonnost za posledních 5 domácích a 5 venkovních zápasů
        last5_home = df[df['HomeTeam'] == home_team].sort_values('Date').tail(5)
        last5_away = df[df['AwayTeam'] == away_team].sort_values('Date').tail(5)

        home_attack = (last5_home[home_col].mean() + season_df[season_df['HomeTeam'] == home_team][home_col].mean()) / 2 / avg_home
        away_defense = (last5_away[home_col].mean() + season_df[season_df['HomeTeam'] == away_team][home_col].mean()) / 2 / avg_home

        away_attack = (last5_away[away_col].mean() + season_df[season_df['AwayTeam'] == away_team][away_col].mean()) / 2 / avg_away
        home_defense = (last5_home[away_col].mean() + season_df[season_df['AwayTeam'] == home_team][away_col].mean()) / 2 / avg_away

        expected_home = avg_home * home_attack * away_defense
        expected_away = avg_away * away_attack * home_defense

        output[stat_name] = {'Home': round(expected_home, 2), 'Away': round(expected_away, 2)}

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

    def get_expected_goals(sub_df):
        league_avg_home_goals = sub_df['FTHG'].mean()
        league_avg_away_goals = sub_df['FTAG'].mean()

        home_attack = sub_df[sub_df['HomeTeam'] == home_team]['FTHG'].mean() / league_avg_home_goals
        home_defense = sub_df[sub_df['AwayTeam'] == home_team]['FTAG'].mean() / league_avg_away_goals
        away_attack = sub_df[sub_df['AwayTeam'] == away_team]['FTAG'].mean() / league_avg_away_goals
        away_defense = sub_df[sub_df['HomeTeam'] == away_team]['FTHG'].mean() / league_avg_home_goals

        expected_home = league_avg_home_goals * home_attack * away_defense
        expected_away = league_avg_away_goals * away_attack * home_defense

        return expected_home, expected_away

    # Historická data
    hist_home, hist_away = get_expected_goals(historical_df)

    # Aktuální sezóna
    season_home, season_away = get_expected_goals(season_df)

    # Posledních 5 zápasů
    last5_home_df = df[df['HomeTeam'] == home_team].tail(5)
    last5_away_df = df[df['AwayTeam'] == away_team].tail(5)
    last5_df = pd.concat([last5_home_df, last5_away_df])
    last5_home, last5_away = get_expected_goals(last5_df)

    expected_home = 0.3 * hist_home + 0.4 * season_home + 0.3 * last5_home
    expected_away = 0.3 * hist_away + 0.4 * season_away + 0.3 * last5_away

    return round(expected_home, 2), round(expected_away, 2)


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

    output = {}
    for stat_name, (home_col, away_col) in stat_columns.items():
        if home_col not in df.columns or away_col not in df.columns:
            output[stat_name] = {'Home': 0.0, 'Away': 0.0}
            continue

        hist_home, hist_away = expected_for_stat(historical_df, home_col, away_col)
        season_home, season_away = expected_for_stat(season_df, home_col, away_col)

        last5_home_df = df[df['HomeTeam'] == home_team].sort_values('Date').tail(5)
        last5_away_df = df[df['AwayTeam'] == away_team].sort_values('Date').tail(5)
        last5_df = pd.concat([last5_home_df, last5_away_df])
        last5_home, last5_away = expected_for_stat(last5_df, home_col, away_col)

        final_home = 0.3 * hist_home + 0.4 * season_home + 0.3 * last5_home
        final_away = 0.3 * hist_away + 0.4 * season_away + 0.3 * last5_away

        output[stat_name] = {
            'Home': round(final_home, 2),
            'Away': round(final_away, 2)
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

    print("⚠️ HISTORICAL:", len(historical_df))
    print("📅 SEASON:", len(season_df))

    def get_expected_goals(sub_df):
        league_avg_home_goals = sub_df['FTHG'].mean()
        league_avg_away_goals = sub_df['FTAG'].mean()

        if pd.isna(league_avg_home_goals) or pd.isna(league_avg_away_goals):
            return 1.0, 1.0

        def safe_mean(filter_df, col):
            return filter_df[col].mean() if not filter_df.empty else league_avg_home_goals

        home_attack = safe_mean(sub_df[sub_df['HomeTeam'] == home_team], 'FTHG') / league_avg_home_goals
        home_defense = safe_mean(sub_df[sub_df['AwayTeam'] == home_team], 'FTAG') / league_avg_away_goals
        away_attack = safe_mean(sub_df[sub_df['AwayTeam'] == away_team], 'FTAG') / league_avg_away_goals
        away_defense = safe_mean(sub_df[sub_df['HomeTeam'] == away_team], 'FTHG') / league_avg_home_goals

        expected_home = league_avg_home_goals * home_attack * away_defense
        expected_away = league_avg_away_goals * away_attack * home_defense

        return expected_home, expected_away

    # Historická data
    hist_home, hist_away = get_expected_goals(historical_df)

    # Aktuální sezóna
    season_home, season_away = get_expected_goals(season_df)

    # Posledních 5 zápasů
    last5_home_df = df[df['HomeTeam'] == home_team].tail(5)
    last5_away_df = df[df['AwayTeam'] == away_team].tail(5)
    last5_df = pd.concat([last5_home_df, last5_away_df])
    print("🔥 LAST 5:", len(last5_df))
    last5_home, last5_away = get_expected_goals(last5_df)

    expected_home = 0.3 * hist_home + 0.4 * season_home + 0.3 * last5_home
    expected_away = 0.3 * hist_away + 0.4 * season_away + 0.3 * last5_away

    return round(expected_home, 2), round(expected_away, 2)

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

def calculate_pseudo_xg(df, team, is_home=True):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    side = 'Home' if is_home else 'Away'
    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    goals_col = 'FTHG' if is_home else 'FTAG'
    shots_col = 'HS' if is_home else 'AS'
    shots_on_target_col = 'HST' if is_home else 'AST'

    team_df = df[df[team_col] == team]
    if team_df.empty:
        return {"avg_xG": 0, "xG_last5": 0, "conversion_rate": 0, "shots_on_target_ratio": 0}

    # Sezónní průměry
    avg_goals = team_df[goals_col].mean()
    avg_shots = team_df[shots_col].mean()
    avg_sot = team_df[shots_on_target_col].mean()

    conversion_rate = avg_goals / avg_sot if avg_sot > 0 else 0
    sot_ratio = avg_sot / avg_shots if avg_shots > 0 else 0
    pseudo_xG = avg_sot * conversion_rate

    # Posledních 5 zápasů
    last5 = team_df.tail(5)
    last5_goals = last5[goals_col].mean()
    last5_sot = last5[shots_on_target_col].mean()
    last5_conversion = last5_goals / last5_sot if last5_sot > 0 else 0
    xg_last5 = last5_sot * last5_conversion

    return {
        "avg_xG": round(pseudo_xG, 2),
        "xG_last5": round(xg_last5, 2),
        "conversion_rate": round(conversion_rate, 2),
        "shots_on_target_ratio": round(sot_ratio, 2)
    }

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
        'Home xP': round(home_xp, 2),
        'Away xP': round(away_xp, 2)
    }