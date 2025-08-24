import pandas as pd
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from sections.match_prediction_section import load_upcoming_xg

def test_upcoming_xg_data_integrity():
    df = load_upcoming_xg()
    # Date column should be datetime-like so comparisons by date work reliably
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])
    # Ensure expected match exists and nonexistent match does not
    d1 = df[df['LeagueCode'] == 'D1']
    assert ((d1['Home Team'] == 'Mainz') & (d1['Away Team'] == 'FC Koln')).any()
    assert not ((d1['Home Team'] == 'Mainz') & (d1['Away Team'] == 'Hoffenheim')).any()
