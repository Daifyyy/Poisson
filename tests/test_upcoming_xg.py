import pandas as pd
import pathlib
import sys
import glob

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from sections.match_prediction_section import load_upcoming_xg, TEAM_NAME_MAP

def test_upcoming_xg_data_integrity():
    df = load_upcoming_xg()
    # Date column should be datetime-like so comparisons by date work reliably
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])
    # Ensure expected match exists and nonexistent match does not
    d1 = df[df['LeagueCode'] == 'D1']
    assert ((d1['Home Team'] == 'Mainz') & (d1['Away Team'] == 'FC Koln')).any()
    assert not ((d1['Home Team'] == 'Mainz') & (d1['Away Team'] == 'Hoffenheim')).any()


def test_team_name_normalization():
    df = load_upcoming_xg()

    # All shorthand team names from the workbook should be replaced by the
    # canonical names from our datasets.
    teams = pd.unique(df[['Home Team', 'Away Team']].values.ravel())
    for shorthand, canonical in TEAM_NAME_MAP.items():
        assert shorthand not in teams
        assert canonical in teams

    # Additionally ensure the remaining team names all exist in the dataset
    # league files so future differences are detected.
    dataset_teams = set()
    for path in glob.glob('data/*_combined_full_updated.csv'):
        csv = pd.read_csv(path, usecols=['HomeTeam', 'AwayTeam'])
        dataset_teams.update(csv['HomeTeam'].unique())
        dataset_teams.update(csv['AwayTeam'].unique())

    assert set(teams).issubset(dataset_teams)
