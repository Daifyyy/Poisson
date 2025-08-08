import sys
import pathlib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils.poisson_utils.stats import aggregate_team_stats


def test_aggregate_team_stats_handles_missing_columns():
    df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'HomeTeam': ['A', 'B'],
        'AwayTeam': ['B', 'A'],
        'FTHG': [1, 2],
        'FTAG': [0, 1],
        'HS': [5, 6],
        'AS': [3, 4],
        'HST': [2, 3],
        'AST': [1, 2],
        'HC': [1, 2],
        'AC': [3, 4],
        # Columns HY, AY, HR, AR, HF, AF intentionally omitted
    })

    stats = aggregate_team_stats(df)

    assert 'Žluté' in stats.columns
    assert 'Červené' in stats.columns
    assert stats.loc['A', 'Žluté'] == 0
    assert stats.loc['B', 'Červené'] == 0
