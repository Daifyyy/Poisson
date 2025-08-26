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
    časy poslední úpravy a velikosti obou souborů.  Novější nebo velikostně
    odlišný ``.csv`` přepíše cache v ``.parquet``.  Volitelným parametrem
    ``force_refresh`` lze vynutit načtení z ``.csv`` bez ohledu na tyto
    kontroly.
    """
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
            elif (
                csv_stat.st_mtime == parquet_stat.st_mtime
                and csv_stat.st_size != parquet_stat.st_size
            ):
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

    # ``df`` may be empty (e.g. when only future fixtures are present).  In
    # that case we fall back to a season start derived from today's date so the
    # function returns sensible defaults instead of raising ``IndexError``.
    if dates.empty:
        latest_date = today
    else:
        latest_date = dates.iloc[-1]

    if not dates.empty:
        date_diffs = dates.diff().fillna(pd.Timedelta(days=0))
        season_breaks = dates[date_diffs > pd.Timedelta(days=gap_days)]
    else:
        season_breaks = pd.Series([], dtype="datetime64[ns]")

    if not season_breaks.empty:
        season_start = season_breaks.iloc[-1]
    else:
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


def ensure_min_season_matches(
    df: pd.DataFrame,
    season_df: pd.DataFrame,
    season_start: pd.Timestamp,
    teams: list[str],
    min_required: int = 10,
    fallback_n: int = 10,
) -> pd.DataFrame:
    """Doplní zápasy z minulé sezony, pokud je aktuální vzorek malý.

    Pokud má některý tým v ``season_df`` méně než ``min_required`` zápasů,
    přidá se mu posledních ``fallback_n`` utkání před ``season_start``.
    """

    result = season_df.copy()
    for team in teams:
        # počítáme pouze zápasy z aktuální sezóny, abychom zabránili tomu, že
        # zápasy přidané pro jiný tým uměle navýší počet utkání tohoto týmu
        season_mask = (season_df["HomeTeam"] == team) | (
            season_df["AwayTeam"] == team
        )
        if season_mask.sum() < min_required:
            prev = df[
                (df["Date"] < season_start)
                & ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
            ].sort_values("Date").tail(fallback_n)
            result = pd.concat([result, prev], ignore_index=True)

    return (
        result.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
        .sort_values("Date")
    )


def load_cup_matches(team_league_map: dict[str, str], data_dir: str | Path = "data") -> pd.DataFrame:
    """Load cup fixtures and map teams to their domestic leagues.

    Parameters
    ----------
    team_league_map : dict
        Mapping of team name to domestic league code. Typically built from
        league CSV files.
    data_dir : str or Path, optional
        Directory containing cup CSV files. Defaults to ``"data"``.

    Returns
    -------
    pd.DataFrame
        Cup matches with columns ``Date``, ``HomeTeam``, ``AwayTeam``, ``FTHG``
        and ``FTAG``. Only rows where both teams have a known league are
        returned. Additional columns ``HomeLeague`` and ``AwayLeague``
        indicate the mapped domestic leagues.
    """

    data_dir = Path(data_dir)
    cup_files = [
        p
        for p in data_dir.glob("*_combined_full.csv")
        if not p.name.endswith("_updated.csv")
    ]
    frames = []
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
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "HomeLeague",
                "AwayLeague",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def load_cup_team_stats(
    team_league_map: dict[str, str],
    data_dir: str | Path = "data",
    matches_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate cup matches into per-team statistics.

    Parameters
    ----------
    team_league_map : dict
        Mapping of team name to domestic league code. Typically built from
        league CSV files.
    data_dir : str or Path, optional
        Directory containing cup CSV files. Defaults to ``"data"``. Ignored
        when ``matches_df`` is provided.
    matches_df : pd.DataFrame, optional
        Pre-loaded cup matches as returned by :func:`load_cup_matches`. Passing
        this avoids reading the CSV files again.

    Returns
    -------
    pd.DataFrame
        Table with columns ``league``, ``team``, ``matches``, ``goals_for``,
        ``goals_against``, ``xg_for`` and ``xg_against``. Shot metrics are set
        to zero because FBref cup pages do not provide them. The ``xg`` values
        are approximated by the goal counts so that downstream calculations can
        still rely on xG columns.
    """

    matches = (
        matches_df.copy()
        if matches_df is not None
        else load_cup_matches(team_league_map, data_dir)
    )
    if matches.empty:
        return pd.DataFrame(
            columns=[
                "league",
                "team",
                "matches",
                "goals_for",
                "goals_against",
                "xg_for",
                "xg_against",
                "shots_for",
                "shots_against",
            ]
        )

    home = matches.rename(
        columns={
            "HomeTeam": "team",
            "FTHG": "goals_for",
            "FTAG": "goals_against",
            "HomeLeague": "league",
        }
    )
    away = matches.rename(
        columns={
            "AwayTeam": "team",
            "FTAG": "goals_for",
            "FTHG": "goals_against",
            "AwayLeague": "league",
        }
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

    agg["xg_for"] = agg["goals_for"]
    agg["xg_against"] = agg["goals_against"]
    agg["shots_for"] = 0
    agg["shots_against"] = 0
    return agg
