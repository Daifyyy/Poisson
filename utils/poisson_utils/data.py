from pathlib import Path

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


def load_data(file_path: str, *, force_refresh: bool = False) -> pd.DataFrame:
    """Načte CSV soubor a připraví ho.

    Pokud existuje soubor ``.parquet`` se stejným názvem, funkce porovná
    časy poslední úpravy a velikosti obou souborů. Novější nebo velikostně
    odlišný ``.csv`` přepíše cache v ``.parquet``. Volitelným parametrem
    ``force_refresh`` lze vynutit načtení z ``.csv`` bez ohledu na tyto
    kontroly.
    """
    numeric_columns = [
        "FTHG", "FTAG",
        "HS", "AS",
        "HST", "AST",
        "HC", "AC",
        "HY", "AY",
        "HR", "AR",
        "HF", "AF",
    ]

    file_path = Path(file_path)
    parquet_path = file_path.with_suffix(".parquet")

    def _read_csv() -> pd.DataFrame:
        dtype_mapping = {col: "Int64" for col in numeric_columns}
        df_csv = pd.read_csv(file_path, dtype=dtype_mapping, parse_dates=["Date"])
        df_csv.to_parquet(parquet_path, index=False)
        return df_csv

    if parquet_path.exists() and not force_refresh:
        if file_path.exists():
            csv_stat = file_path.stat()
            parquet_stat = parquet_path.stat()
            if csv_stat.st_mtime > parquet_stat.st_mtime:
                df = _read_csv()
            elif (csv_stat.st_mtime == parquet_stat.st_mtime
                  and csv_stat.st_size != parquet_stat.st_size):
                df = _read_csv()
            else:
                df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)
    elif force_refresh and file_path.exists():
        df = _read_csv()
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = _read_csv()

    df = prepare_df(df)
    required_columns = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG",
        "HS", "AS",
        "HST", "AST",
        "HC", "AC",
        "FTR",
        "HY", "AY",
        "HR", "AR",
        "HF", "AF",
    ]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

    return df


def detect_current_season(
    df: pd.DataFrame,
    *,
    start_month: int = 8,
    gap_days: int = 30,
    prepared: bool = False,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Vrátí zápasy aktuální sezóny a detekovaný začátek sezóny.

    Detekce: pokud v datech existují pauzy > ``gap_days``, bere se start
    poslední takové pauzy. Jinak fallback na 1. ``start_month`` dle
    posledního odehraného zápasu. Budoucí zápasy ignorujeme.
    """
    if not prepared:
        df = prepare_df(df)

    today = pd.Timestamp.today().normalize()
    df = df[df["Date"] <= today]

    dates = df["Date"].drop_duplicates().sort_values().reset_index(drop=True)
    latest_date = dates.iloc[-1] if not dates.empty else today

    if not dates.empty:
        date_diffs = dates.diff().fillna(pd.Timedelta(days=0))
        season_breaks = dates[date_diffs > pd.Timedelta(days=gap_days)]
    else:
        season_breaks = pd.Series([], dtype="datetime64[ns]")

    if not season_breaks.empty:
        season_start = season_breaks.iloc[-1]
    else:
        if latest_date.month >= start_month:
            season_start = pd.Timestamp(year=latest_date.year, month=start_month, day=1)
        else:
            season_start = pd.Timestamp(year=latest_date.year - 1, month=start_month, day=1)

    season_df = df[df["Date"] >= season_start]
    return season_df, season_start


def get_last_n_matches(df: pd.DataFrame, team: str, role: str = "both", n: int = 10) -> pd.DataFrame:
    if role == "home":
        matches = df[df['HomeTeam'] == team]
    elif role == "away":
        matches = df[df['AwayTeam'] == team]
    else:
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    return matches.sort_values("Date").tail(n)


def ensure_min_season_matches(
    df: pd.DataFrame,
    season_df: pd.DataFrame,
    season_start: pd.Timestamp,
    teams: list[str],
    min_required: int = 10,
    fallback_n: int = 10,
) -> pd.DataFrame:
    """Doplní zápasy z minulé sezony, pokud je aktuální vzorek malý.

    Pro každý tým z ``teams`` zkontroluje počet zápasů v ``season_df``.
    Pokud je menší než ``min_required``, přidá posledních ``fallback_n``
    jeho utkání před ``season_start``. Duplicitní zápasy (Date, HomeTeam, AwayTeam)
    se odstraní.
    """
    result = season_df.copy()

    for team in teams:
        # Počet se bere z původního season_df, aby přidané zápasy jiných týmů
        # neovlivnily rozhodnutí pro tento tým.
        season_count = ((season_df["HomeTeam"] == team) | (season_df["AwayTeam"] == team)).sum()
        if season_count < min_required:
            prev = (
                df[(df["Date"] < season_start)
                   & ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))]
                .sort_values("Date")
                .tail(fallback_n)
            )
            if not prev.empty:
                result = pd.concat([result, prev], ignore_index=True)

    return (
        result.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
              .sort_values("Date")
    )


def load_cup_matches(team_league_map: dict[str, str], data_dir: str | Path = "data") -> pd.DataFrame:
    """Načte pohárové zápasy a namapuje týmy na jejich ligy."""
    data_dir = Path(data_dir)
    cup_files = [
        p for p in data_dir.glob("*_combined_full.csv")
        if not p.name.endswith("_updated.csv")
    ]
    frames: list[pd.DataFrame] = []
    for path in cup_files:
        df = pd.read_csv(path)
        df = prepare_df(df)
        df["HomeLeague"] = df["HomeTeam"].map(team_league_map)
        df["AwayLeague"] = df["AwayTeam"].map(team_league_map)
        df = df[df["HomeLeague"].notna() & df["AwayLeague"].notna()]
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                "HomeLeague", "AwayLeague",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def load_cup_team_stats(
    team_league_map: dict[str, str],
    data_dir: str | Path = "data",
    matches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggreguje pohárové zápasy do týmových statistik."""
    matches = matches_df.copy() if matches_df is not None else load_cup_matches(team_league_map, data_dir)
    if matches.empty:
        return pd.DataFrame(
            columns=[
                "league", "team", "matches",
                "goals_for", "goals_against",
                "xg_for", "xg_against",
                "shots_for", "shots_against",
            ]
        )

    home = matches.rename(
        columns={"HomeTeam": "team", "FTHG": "goals_for", "FTAG": "goals_against", "HomeLeague": "league"}
    )
    away = matches.rename(
        columns={"AwayTeam": "team", "FTAG": "goals_for", "FTHG": "goals_against", "AwayLeague": "league"}
    )
    combined = pd.concat([home, away], ignore_index=True)

    agg = (
        combined.groupby(["league", "team"])
        .agg(
            matches=("goals_for", "size"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
        )
        .reset_index()
    )

    # FBref poháry nemají střelecké metriky – nastavíme 0 a xG ≈ góly.
    agg["xg_for"] = agg["goals_for"]
    agg["xg_against"] = agg["goals_against"]
    agg["shots_for"] = 0
    agg["shots_against"] = 0
    return agg
