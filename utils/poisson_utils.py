
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
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={'label': 'Pravděpodobnost (%)'})
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

def expected_team_stats_weighted_by_elo(df, home_team, away_team, stat_columns, elo_dict):
    import numpy as np
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

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



# def expected_goals_weighted_final(df, home_team, away_team):
#     df = df.copy()
#     df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
#     df = df.dropna(subset=['Date'])
#     df = df.sort_values('Date')

#     latest_date = df['Date'].max()
#     one_year_ago = latest_date - pd.Timedelta(days=365)

#     season_df = df[df['Date'] >= one_year_ago]
#     historical_df = df[df['Date'] < one_year_ago]

#     def calculate_team_strengths(df, fallback_df=None):
#         all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
#         strengths = {}

#         for team in all_teams:
#             home_attack = df[df['HomeTeam'] == team]['FTHG'].mean()
#             away_attack = df[df['AwayTeam'] == team]['FTAG'].mean()
#             home_defense = df[df['HomeTeam'] == team]['FTAG'].mean()
#             away_defense = df[df['AwayTeam'] == team]['FTHG'].mean()

#             if all(np.isnan(v) for v in [home_attack, away_attack, home_defense, away_defense]) and fallback_df is not None:
#                 home_attack = fallback_df[fallback_df['HomeTeam'] == team]['FTHG'].mean()
#                 away_attack = fallback_df[fallback_df['AwayTeam'] == team]['FTAG'].mean()
#                 home_defense = fallback_df[fallback_df['HomeTeam'] == team]['FTAG'].mean()
#                 away_defense = fallback_df[fallback_df['AwayTeam'] == team]['FTHG'].mean()

#             strengths[team] = {
#                 'attack': np.nanmean([home_attack, away_attack]),
#                 'defense': np.nanmean([home_defense, away_defense])
#             }

#         return strengths

#     def get_expected(df_subset, strengths):
#         league_avg_home = df_subset['FTHG'].mean()
#         league_avg_away = df_subset['FTAG'].mean()

#         if np.isnan(league_avg_home) or np.isnan(league_avg_away):
#             return np.nan, np.nan

#         ha = np.nanmean([
#             df_subset[df_subset['HomeTeam'] == home_team]['FTHG'].mean(),
#             df_subset[df_subset['AwayTeam'] == home_team]['FTAG'].mean()
#         ]) / league_avg_home

#         hd = np.nanmean([
#             df_subset[df_subset['AwayTeam'] == home_team]['FTAG'].mean(),
#             df_subset[df_subset['HomeTeam'] == home_team]['FTAG'].mean()
#         ]) / league_avg_away

#         aa = np.nanmean([
#             df_subset[df_subset['AwayTeam'] == away_team]['FTAG'].mean(),
#             df_subset[df_subset['HomeTeam'] == away_team]['FTHG'].mean()
#         ]) / league_avg_away

#         ad = np.nanmean([
#             df_subset[df_subset['HomeTeam'] == away_team]['FTHG'].mean(),
#             df_subset[df_subset['AwayTeam'] == away_team]['FTHG'].mean()
#         ]) / league_avg_home

#         if any(pd.isna(x) for x in [ha, hd, aa, ad]):
#             return np.nan, np.nan

#         expected_home = league_avg_home * ha * ad
#         expected_away = league_avg_away * aa * hd
#         return expected_home, expected_away

#     hist_strengths = calculate_team_strengths(historical_df)
#     season_strengths = calculate_team_strengths(season_df, fallback_df=df)

#     hist_home, hist_away = get_expected(historical_df, hist_strengths)
#     season_home, season_away = get_expected(season_df, season_strengths)

#     # Posledních 5 zápasů celkem (home i away)
#     last5_home = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].sort_values('Date').tail(5)
#     last5_away = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].sort_values('Date').tail(5)
#     last5_df = pd.concat([last5_home, last5_away])

#     # Zde se výpočet obejde bez síly soupeřů
#     def expected_from_last5(team, is_home):
#         team_matches = last5_df[(last5_df['HomeTeam'] == team) | (last5_df['AwayTeam'] == team)]
#         goals_scored = []
#         goals_conceded = []

#         for _, row in team_matches.iterrows():
#             if row['HomeTeam'] == team:
#                 goals_scored.append(row['FTHG'])
#                 goals_conceded.append(row['FTAG'])
#             else:
#                 goals_scored.append(row['FTAG'])
#                 goals_conceded.append(row['FTHG'])

#         return np.mean(goals_scored), np.mean(goals_conceded)

#     last5_home_exp, _ = expected_from_last5(home_team, is_home=True)
#     _, last5_away_exp = expected_from_last5(away_team, is_home=False)

#     def any_nan(*args):
#         return any(pd.isna(x) for x in args)

#     if any_nan(hist_home, season_home, last5_home_exp, hist_away, season_away, last5_away_exp):
#         raise ValueError(f"❌ Nedostatek dat pro predikci mezi {home_team} a {away_team}.")

#     final_home = 0.3 * hist_home + 0.4 * season_home + 0.3 * last5_home_exp
#     final_away = 0.3 * hist_away + 0.4 * season_away + 0.3 * last5_away_exp

#     return round(final_home, 1), round(final_away, 1)

def expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict):
    import numpy as np
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

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

def calculate_pseudo_xg(df, team):
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

def extended_home_away_opponent_form(df, team):
    import numpy as np
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    # Rozdělení podle síly soupeřů
    team_avg_goals = {}
    for t in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        home_g = df[df['HomeTeam'] == t]['FTHG'].mean()
        away_g = df[df['AwayTeam'] == t]['FTAG'].mean()
        team_avg_goals[t] = np.nanmean([home_g, away_g])

    sorted_teams = sorted(team_avg_goals.items(), key=lambda x: x[1], reverse=True)
    top = set([t for t, _ in sorted_teams[:int(0.3 * len(sorted_teams))]])
    bottom = set([t for t, _ in sorted_teams[-int(0.3 * len(sorted_teams)):]])
    middle = set(team_avg_goals.keys()) - top - bottom

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
        subset["points"] = subset.apply(lambda r: 3 if r["FTHG"] > r["FTAG"] and is_home else 3 if r["FTAG"] > r["FTHG"] and not is_home else 1 if r["FTHG"] == r["FTAG"] else 0, axis=1)

        result[loc]["vs_strong"] = summarize(subset[subset["opponent"].isin(top)])
        result[loc]["vs_average"] = summarize(subset[subset["opponent"].isin(middle)])
        result[loc]["vs_weak"] = summarize(subset[subset["opponent"].isin(bottom)])

    return result

def merged_home_away_opponent_form(df, team):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    # Rozdělení týmů podle síly
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
            return {"Z": 0, "G": 0, "OG": 0, "S": 0, "SOT": 0, "xG": 0, "PTS": 0}
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
        return {
            "Z": len(matches),
            "G": round(goals_for.mean(), 2),
            "OG": round(goals_against.mean(), 2),
            "S": round(shots.mean(), 1),
            "SOT": round(sot.mean(), 1),
            "xG": xg,
            "PTS": round(points, 2)
        }

    def combine_metrics(strong, average, weak, label):
        return {
            label: {
                "Zápasy": f"{strong['Z']} / {average['Z']}",
                "Góly": f"{strong['G']} / {average['G']}",
                "Obdržené": f"{strong['OG']} / {average['OG']}",
                "Střely": f"{strong['S']} / {average['S']}",
                "Na branku": f"{strong['SOT']} / {average['SOT']}",
                "xG": f"{strong['xG']} / {average['xG']}",
                "Body/zápas": f"{strong['PTS']} / {average['PTS']}"
            }
        }

    def generate_table():
        result = {}
        for label, group in [("💪 Silní", top), ("⚖️ Průměrní", middle), ("🪶 Slabí", bottom)]:
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
                "Body/zápas": f"{home_stats['PTS']} / {away_stats['PTS']}"
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
    df = df.copy()
    df['OppELO'] = df[opp_col].map(elo_dict)
    df['EloDiff'] = abs(df['OppELO'] - opponent_elo)
    matches = df[df[team_col] == team].sort_values('EloDiff').head(last_n)

    if matches.empty:
        matches = df[df[team_col] == team].sort_values("Date", ascending=False).head(last_n)

    if matches.empty:
        return {"tempo": 0, "percentile": 0, "rating": "N/A", "imbalance": 0.0, "imbalance_type": "N/A"}

    shots = matches['HS'] if is_home else matches['AS']
    corners = matches['HC'] if is_home else matches['AC']
    fouls = matches['HF'] if is_home else matches['AF']
    tempo_index = (shots + corners + fouls).mean()

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

    return {
        "tempo": round(tempo_index, 1),
        "percentile": percentile,
        "rating": rating,
        "imbalance": round(imbalance, 2),
        "imbalance_type": imbalance_type
    }
    
def classify_team_strength(df, team):
    df = df.copy()
    avg_goals = {}
    for t in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        home_avg = df[df['HomeTeam'] == t]['FTHG'].mean()
        away_avg = df[df['AwayTeam'] == t]['FTAG'].mean()
        avg_goals[t] = np.nanmean([home_avg, away_avg])

    sorted_teams = sorted(avg_goals.items(), key=lambda x: x[1], reverse=True)
    total = len(sorted_teams)
    top = set(t for t, _ in sorted_teams[:int(total * 0.3)])
    bottom = set(t for t, _ in sorted_teams[-int(total * 0.3):])

    if team in top:
        return "💪 Silný"
    elif team in bottom:
        return "🪶 Slabý"
    else:
        return "⚖️ Průměrný"