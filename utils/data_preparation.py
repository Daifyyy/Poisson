import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['FTHG', 'FTAG', 'Date'])
    df = df.sort_values('Date')
    return df


def prepare_df(df):
    """Základní úprava dat: kopírování, převod datumu, odstranění nevalidních řádků, seřazení podle data."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    return df


def detect_current_season(df):
    df = prepare_df(df)
    dates = df['Date'].drop_duplicates().sort_values().reset_index(drop=True)
    date_diffs = dates.diff().fillna(pd.Timedelta(days=0))
    season_breaks = dates[date_diffs > pd.Timedelta(days=30)].reset_index(drop=True)
    if not season_breaks.empty:
        season_start = season_breaks.iloc[-1]
    else:
        season_start = dates.iloc[0]
    print(season_start)
    return df[df['Date'] >= season_start], season_start
