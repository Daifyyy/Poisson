import pandas as pd


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Základní úprava dat: kopírování, převod datumu, odstranění nevalidních řádků, seřazení podle data."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]
    df = df[df["AwayTeam"].astype(str).str.strip() != ""]
    df = df.sort_values('Date')
    return df


def load_data(file_path: str) -> pd.DataFrame:
    """Načte CSV soubor a připraví ho."""
    df = pd.read_csv(file_path)
    df = prepare_df(df)
    required_columns = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "FTR",
        "HY",
        "AY",
        "HR",
        "AR",
        "HF",
        "AF",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    numeric_columns = [
        "FTHG",
        "FTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
        "HF",
        "AF",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def detect_current_season(
    df: pd.DataFrame,
    *,
    start_month: int = 8,
    gap_days: int = 30,
    prepared: bool = False,
) -> tuple:
    """Return matches belonging to the current season.

    The function tries to infer the start of the ongoing season in a more
    flexible way than simply assuming an ``1 August`` cut-off.  It first scans
    match dates for long gaps (``gap_days``) and treats the match following the
    last such break as the season start.  If no significant gap is found, it
    falls back to the conventional start ``start_month`` relative to the latest
    played match.  Future fixtures are ignored so that upcoming games do not
    shift the detected season prematurely.

    Parameters
    ----------
    df : pd.DataFrame
        League match data containing a ``Date`` column.
    start_month : int, optional
        Month used as a fallback start when no large breaks are present.
    gap_days : int, optional
        Minimum number of days considered a break between seasons.
    prepared : bool, optional
        Set to ``True`` if ``df`` has already been processed by
        :func:`prepare_df` to avoid running it twice.

    Returns
    -------
    tuple
        ``(season_df, season_start)`` where ``season_df`` contains only matches
        from the detected season.
    """

    if not prepared:
        df = prepare_df(df)

    # Work only with past matches to avoid jumping to the next season because
    # of fixtures far in the future.
    today = pd.Timestamp.today().normalize()
    df = df[df["Date"] <= today]

    dates = df["Date"].drop_duplicates().sort_values().reset_index(drop=True)
    date_diffs = dates.diff().fillna(pd.Timedelta(days=0))
    season_breaks = dates[date_diffs > pd.Timedelta(days=gap_days)]

    if not season_breaks.empty:
        season_start = season_breaks.iloc[-1]
    else:
        latest_date = dates.iloc[-1]
        if latest_date.month >= start_month:
            season_start = pd.Timestamp(
                year=latest_date.year, month=start_month, day=1
            )
        else:
            season_start = pd.Timestamp(
                year=latest_date.year - 1, month=start_month, day=1
            )

    season_df = df[df["Date"] >= season_start]
    return season_df, season_start


def get_last_n_matches(df, team, role="both", n=10):
    if role == "home":
        matches = df[df['HomeTeam'] == team]
    elif role == "away":
        matches = df[df['AwayTeam'] == team]
    else:
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    return matches.sort_values("Date").tail(n)
