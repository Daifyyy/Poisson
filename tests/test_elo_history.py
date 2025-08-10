import sys
import pathlib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils.poisson_utils.elo import elo_history


def test_elo_history_updates_per_match():
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15']),
        'HomeTeam': ['A', 'B', 'A'],
        'AwayTeam': ['B', 'A', 'C'],
        'FTHG': [1, 2, 0],
        'FTAG': [0, 1, 1],
    })

    history = elo_history(df, 'A')
    assert len(history) == 3
    assert history['Date'].tolist() == sorted(history['Date'].tolist())

    # manual incremental calculation
    df_sorted = df.sort_values('Date')
    teams = pd.concat([df_sorted['HomeTeam'], df_sorted['AwayTeam']]).unique()
    elo = {t: 1500 for t in teams}
    expected = []
    for _, row in df_sorted.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        hg, ag = row['FTHG'], row['FTAG']
        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 / (1 + 10 ** ((elo[home] - elo[away]) / 400))
        if hg > ag:
            res_home, res_away = 1, 0
        elif hg < ag:
            res_home, res_away = 0, 1
        else:
            res_home = res_away = 0.5
        elo[home] += 20 * (res_home - exp_home)
        elo[away] += 20 * (res_away - exp_away)
        if home == 'A' or away == 'A':
            expected.append(elo['A'])

    assert history['ELO'].tolist() == expected
