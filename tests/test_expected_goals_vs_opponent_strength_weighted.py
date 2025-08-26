import pandas as pd
import numpy as np

from utils.poisson_utils.data import prepare_df
from utils.poisson_utils.team_analysis import expected_goals_vs_opponent_strength_weighted


def naive_expected_goals_vs_opponent_strength_weighted(df, team, opponent, elo_dict, is_home=True, n=20):
    df = prepare_df(df).sort_values("Date")
    team_matches = df[df['HomeTeam'] == team] if is_home else df[df['AwayTeam'] == team]
    team_matches = team_matches.tail(n)
    if team_matches.empty:
        return 1.0

    opp_col = 'AwayTeam' if is_home else 'HomeTeam'
    gf_col = 'FTHG' if is_home else 'FTAG'

    team_matches = team_matches.copy()
    team_matches["Opponent"] = team_matches[opp_col]
    team_matches["EloOpp"] = team_matches["Opponent"].map(elo_dict)
    team_matches = team_matches.dropna(subset=["EloOpp"])

    opp_elo = elo_dict.get(opponent, 1500)
    all_elos = list(elo_dict.values())
    p30 = np.percentile(all_elos, 30)
    p70 = np.percentile(all_elos, 70)

    def classify(e):
        if e <= p30:
            return "weak"
        elif e >= p70:
            return "strong"
        else:
            return "average"

    team_matches["OppStrength"] = team_matches["EloOpp"].apply(classify)
    current_strength = classify(opp_elo)

    gfs = {}
    for group in ["strong", "average", "weak"]:
        sub = team_matches[team_matches["OppStrength"] == group]
        gfs[group] = sub[gf_col].mean() if not sub.empty else 1.0

    weights = {
        "strong": 0.6 if current_strength == "strong" else 0.2,
        "average": 0.6 if current_strength == "average" else 0.2,
        "weak": 0.6 if current_strength == "weak" else 0.2,
    }

    expected = (
        weights["strong"] * gfs["strong"] +
        weights["average"] * gfs["average"] +
        weights["weak"] * gfs["weak"]
    )
    return round(expected, 2)


def test_normalized_expected_goals_reasonable():
    df = pd.DataFrame({
        "Date": [
            "2023-01-01", "2023-01-02", "2023-01-03",
            "2023-02-01", "2023-02-02", "2023-02-03",
        ],
        "HomeTeam": ["A", "A", "A", "B", "D", "F"],
        "AwayTeam": ["StrongTeam", "WeakTeam", "AvgTeam", "C", "E", "G"],
        "FTHG": [1, 3, 2, 1, 2, 0],
        "FTAG": [0, 1, 1, 1, 2, 0],
    })

    elo_dict = {
        "A": 1500,
        "StrongTeam": 1800,
        "WeakTeam": 1100,
        "AvgTeam": 1500,
        "B": 1000,
        "C": 1200,
        "D": 1300,
        "E": 1400,
        "F": 1600,
        "G": 1700,
    }

    old = naive_expected_goals_vs_opponent_strength_weighted(df, "A", "StrongTeam", elo_dict)
    new = expected_goals_vs_opponent_strength_weighted(df, "A", "StrongTeam", elo_dict)

    assert 0 < old < 5
    assert 0 < new < 5
    assert abs(new - old) < 1.0
