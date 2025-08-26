import pandas as pd

from utils.poisson_utils.data import (
    prepare_df,
    detect_current_season,
    ensure_min_season_matches,
)


def test_ensure_min_season_matches_adds_previous_season():
    dates_prev = pd.date_range('2023-02-01', periods=20, freq='7D')
    prev = pd.DataFrame({
        'Date': dates_prev,
        'HomeTeam': ['A'] * 20,
        'AwayTeam': ['B'] * 20,
        'FTHG': 1,
        'FTAG': 1,
    })
    dates_curr = pd.date_range('2023-08-01', periods=3, freq='7D')
    curr = pd.DataFrame({
        'Date': dates_curr,
        'HomeTeam': ['A', 'A', 'A'],
        'AwayTeam': ['B', 'C', 'D'],
        'FTHG': 1,
        'FTAG': 0,
    })
    df = pd.concat([prev, curr], ignore_index=True)
    df = prepare_df(df)
    season_df, season_start = detect_current_season(df, prepared=True)
    augmented = ensure_min_season_matches(df, season_df, season_start, ['A'])
    team_matches = augmented[(augmented['HomeTeam'] == 'A') | (augmented['AwayTeam'] == 'A')]
    assert len(team_matches) == len(curr) + 10


def test_ensure_min_season_matches_handles_multiple_teams():
    dates_prev_a = pd.date_range('2023-02-01', periods=12, freq='7D')
    prev_a = pd.DataFrame({
        'Date': dates_prev_a,
        'HomeTeam': ['A'] * 12,
        'AwayTeam': ['X'] * 12,
        'FTHG': 1,
        'FTAG': 0,
    })
    dates_prev_b = pd.date_range('2023-02-01', periods=12, freq='7D')
    prev_b = pd.DataFrame({
        'Date': dates_prev_b,
        'HomeTeam': ['Y'] * 12,
        'AwayTeam': ['B'] * 12,
        'FTHG': 0,
        'FTAG': 1,
    })
    dates_curr = pd.date_range('2023-08-01', periods=2, freq='7D')
    curr = pd.DataFrame({
        'Date': dates_curr,
        'HomeTeam': ['A', 'B'],
        'AwayTeam': ['B', 'A'],
        'FTHG': [1, 0],
        'FTAG': [0, 1],
    })
    df = pd.concat([prev_a, prev_b, curr], ignore_index=True)
    df = prepare_df(df)
    season_df, season_start = detect_current_season(df, prepared=True)
    augmented = ensure_min_season_matches(df, season_df, season_start, ['A', 'B'])
    count_a = len(augmented[(augmented['HomeTeam'] == 'A') | (augmented['AwayTeam'] == 'A')])
    count_b = len(augmented[(augmented['HomeTeam'] == 'B') | (augmented['AwayTeam'] == 'B')])
    assert count_a == 12
    assert count_b == 12
