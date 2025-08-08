import pandas as pd


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Základní úprava dat: kopírování, převod datumu, odstranění nevalidních řádků, seřazení podle data."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]
    df = df[df["AwayTeam"].astype(str).str.strip() != ""]
    df = df.sort_values('Date')
    return df


def load_data(file_path: str) -> pd.DataFrame:
    """Načte CSV soubor a připraví ho."""
    df = pd.read_csv(file_path)
    df = prepare_df(df)
    required_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def detect_current_season(df: pd.DataFrame) -> tuple:
    """Detekuje aktuální sezónu podle nejnovějšího data."""
    df = prepare_df(df)
    latest_date = df['Date'].max()
    if latest_date.month > 6:
        season_start = pd.Timestamp(year=latest_date.year, month=8, day=1)
    else:
        season_start = pd.Timestamp(year=latest_date.year - 1, month=8, day=1)
    season_df = df[df['Date'] >= season_start]
    return season_df, season_start


def get_last_n_matches(df, team, role="both", n=10):
    if role == "home":
        matches = df[df['HomeTeam'] == team]
    elif role == "away":
        matches = df[df['AwayTeam'] == team]
    else:
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    return matches.sort_values("Date").tail(n)
