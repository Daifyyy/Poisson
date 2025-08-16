import sys
import pathlib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils.poisson_utils.team_analysis import (
    calculate_recent_team_form,
    calculate_expected_and_actual_points,
)


def naive_recent_team_form(df: pd.DataFrame, last_n: int = 5) -> pd.DataFrame:
    from utils.poisson_utils.stats import calculate_points

    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    form_stats = []
    for team in teams:
        recent_matches = (
            df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
            .sort_values("Date")
            .tail(last_n)
        )
        total_points = 0
        for _, row in recent_matches.iterrows():
            is_home = row["HomeTeam"] == team
            total_points += calculate_points(row, is_home)
        form_stats.append({"Tým": team, "Body/zápas": total_points / last_n})
    return pd.DataFrame(form_stats).sort_values("Tým").reset_index(drop=True)


def test_recent_team_form_parity():
    df = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "HomeTeam": ["A", "B", "A", "C"],
            "AwayTeam": ["B", "A", "C", "A"],
            "FTHG": [1, 0, 2, 1],
            "FTAG": [0, 1, 2, 1],
        }
    )

    old = naive_recent_team_form(df, last_n=2)
    new = (
        calculate_recent_team_form(df, last_n=2)[["Tým", "Body/zápas"]]
        .sort_values("Tým")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(old, new)


def naive_expected_and_actual_points(df: pd.DataFrame) -> dict:
    from scipy.stats import poisson

    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    results = {}
    for team in teams:
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]
        all_matches = pd.concat([home, away])
        if all_matches.empty:
            results[team] = {
                "points": 0,
                "points_per_game": 0,
                "matches": 0,
                "expected_points": 0,
            }
            continue

        home_points = sum(
            3 if row["FTHG"] > row["FTAG"] else 1 if row["FTHG"] == row["FTAG"] else 0
            for _, row in home.iterrows()
        )
        away_points = sum(
            3 if row["FTAG"] > row["FTHG"] else 1 if row["FTAG"] == row["FTHG"] else 0
            for _, row in away.iterrows()
        )
        total_points = home_points + away_points
        num_matches = len(home) + len(away)

        xP = 0.0
        for _, row in all_matches.iterrows():
            if row["HomeTeam"] == team:
                xg_for = (row["HST"] / row["HS"]) if row["HS"] > 0 else 0.1
                xg_against = (row["AST"] / row["AS"]) if row["AS"] > 0 else 0.1
                team_is_home = True
            else:
                xg_for = (row["AST"] / row["AS"]) if row["AS"] > 0 else 0.1
                xg_against = (row["HST"] / row["HS"]) if row["HS"] > 0 else 0.1
                team_is_home = False

            max_goals = 6
            for i in range(max_goals):
                for j in range(max_goals):
                    p = poisson.pmf(i, xg_for) * poisson.pmf(j, xg_against)
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
            "points_per_game": round(total_points / num_matches, 2)
            if num_matches > 0
            else 0,
            "matches": num_matches,
            "expected_points": round(xP, 2),
        }

    return results


def test_expected_and_actual_points_parity():
    df = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-02"],
            "HomeTeam": ["A", "B"],
            "AwayTeam": ["B", "A"],
            "FTHG": [1, 0],
            "FTAG": [0, 2],
            "HS": [5, 3],
            "HST": [3, 1],
            "AS": [4, 2],
            "AST": [1, 1],
        }
    )

    old = naive_expected_and_actual_points(df)
    new = calculate_expected_and_actual_points(df)
    assert old == new

