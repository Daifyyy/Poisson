import pandas as pd
import numpy as np
from scipy.stats import poisson

from .core import prepare_df,get_last_n_matches, calculate_points,poisson_over25_probability,expected_goals_vs_similar_elo_weighted 
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

def expected_goals_combined_homeaway_allmatches(df, home_team, away_team, elo_dict,
                                                 weight_homeaway=0.4, weight_all=0.3, weight_elo=0.3):
    df = prepare_df(df)
    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)

    # časová období
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
        val = series.median()
        return val if not np.isnan(val) else default

    def get_home_away_exp(sub, team, is_home):
        if is_home:
            m = sub[sub['HomeTeam'] == team]
            gf = safe_stat(m['FTHG'])
            ga = safe_stat(m['FTAG'])
        else:
            m = sub[sub['AwayTeam'] == team]
            gf = safe_stat(m['FTAG'])
            ga = safe_stat(m['FTHG'])
        return gf, ga

    def get_all_matches_exp(sub, team):
        m = sub[(sub['HomeTeam'] == team) | (sub['AwayTeam'] == team)]
        gf_list, ga_list = [], []
        for _, row in m.iterrows():
            if row['HomeTeam'] == team:
                gf_list.append(row['FTHG'])
                ga_list.append(row['FTAG'])
            else:
                gf_list.append(row['FTAG'])
                ga_list.append(row['FTHG'])
        return safe_stat(pd.Series(gf_list)), safe_stat(pd.Series(ga_list))

    def compute_expected(gf, ga_opp, l_home, l_away):
        return l_home * (gf / l_home) * (ga_opp / l_away)

    def compute_weighted_exp(dfs_home, dfs_away, extractor_home, extractor_away):
        eh, ea = [], []
        for dfh, dfa in zip(dfs_home, dfs_away):
            gf_home, ga_home = extractor_home(dfh)
            gf_away, ga_away = extractor_away(dfa)
            eh.append(compute_expected(gf_home, ga_away, league_avg_home, league_avg_away))
            ea.append(compute_expected(gf_away, ga_home, league_avg_away, league_avg_home))
        weighted_home = 0.15 * eh[0] + 0.5 * eh[1] + 0.35 * eh[2]
        weighted_away = 0.15 * ea[0] + 0.5 * ea[1] + 0.35 * ea[2]
        return weighted_home, weighted_away, eh, ea

    # Výpočty pro přístup 1 (Home/Away only)
    exp_ha_home, exp_ha_away,eh, ea = compute_weighted_exp(
        [df_hist, df_season, df_last5_home],
        [df_hist, df_season, df_last5_away],
        lambda d: get_home_away_exp(d, home_team, True),
        lambda d: get_home_away_exp(d, away_team, False)
    )

    # Výpočty pro přístup 2 (All matches)
    exp_all_home, exp_all_away,eh_all, ea_all = compute_weighted_exp(
        [df_hist, df_season, df_last5_all_home],
        [df_hist, df_season, df_last5_all_away],
        lambda d: get_all_matches_exp(d, home_team),
        lambda d: get_all_matches_exp(d, away_team)
    )

    # Výpočty pro přístup 3 (ELO relevantní + stáří vážené)
    exp_elo_home, exp_elo_away = expected_goals_vs_similar_elo_weighted(df, home_team, away_team, elo_dict)

    # Finální vážená kombinace
    final_home = round(
        weight_homeaway * exp_ha_home +
        weight_all * exp_all_home +
        weight_elo * exp_elo_home, 2)

    final_away = round(
        weight_homeaway * exp_ha_away +
        weight_all * exp_all_away +
        weight_elo * exp_elo_away, 2)

    # Výpis pravděpodobností Over 2.5
    print("🟦 Home/Away-only přístup – Over 2.5:")
    print(f"  Historie:      {poisson_over25_probability(eh[0], ea[0])}%")
    print(f"  Sezóna:        {poisson_over25_probability(eh[1], ea[1])}%")
    print(f"  Posledních 5:  {poisson_over25_probability(eh[2], ea[2])}%")
    print(f"  => Průměr:     {poisson_over25_probability(exp_ha_home, exp_ha_away)}%")

    print("🟧 All matches přístup – Over 2.5:")
    print(f"  Historie:      {poisson_over25_probability(eh_all[0], ea_all[0])}%")
    print(f"  Sezóna:        {poisson_over25_probability(eh_all[1], ea_all[1])}%")
    print(f"  Posledních 5:  {poisson_over25_probability(eh_all[2], ea_all[2])}%")
    print(f"  => Průměr:     {poisson_over25_probability(exp_all_home, exp_all_away)}%")

    print("🧠 Similar ELO přístup – Over 2.5:")
    print(f"  => Výsledek:   {poisson_over25_probability(exp_elo_home, exp_elo_away)}%")

    print("🎯 Finální kombinovaná Over 2.5:")
    print(f"  Výsledek:      {poisson_over25_probability(final_home, final_away)}%")

    return final_home, final_away


