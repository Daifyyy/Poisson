"""Utilities for downloading match data from the FBR API.

The original repository expected a module named ``fbrapi_dataset`` that could
build a training dataset for the Random Forest model. This file provides that
functionality and integrates automatic API key handling so that the key is
generated only once and then cached locally.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import requests

from utils.poisson_utils.xg_sources.fbrapi import get_fbrapi_api_key
from utils.poisson_utils.elo import calculate_elo_ratings


BASE = "https://fbrapi.com"
API_KEY = get_fbrapi_api_key()
HDRS = {"X-API-Key": API_KEY} if API_KEY else {}
SLEEP = 3.3  # ~1 request / 3 s
RETRIES = 3
CACHE_DIR = Path("cache_fbrapi")
CACHE_DIR.mkdir(exist_ok=True)


def _cached_json(filename: str) -> Dict[str, Any] | None:
    p = CACHE_DIR / filename
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(filename: str, data: Dict[str, Any]) -> None:
    (CACHE_DIR / filename).write_text(json.dumps(data), encoding="utf-8")


def _get(path: str, params: Dict[str, Any] | None = None, cache_key: str | None = None) -> Dict[str, Any]:
    if cache_key:
        cached = _cached_json(cache_key)
        if cached is not None:
            return cached

    if not API_KEY:
        raise RuntimeError("Missing FBR API key; call get_fbrapi_api_key() first")

    url = f"{BASE}{path}"
    last_exc: Exception | None = None
    for _ in range(RETRIES):
        try:
            r = requests.get(url, params=params or {}, headers=HDRS, timeout=60)
            if r.status_code == 429:
                time.sleep(max(SLEEP, 3.0))
                continue
            r.raise_for_status()
            data = r.json()
            if cache_key:
                _save_cache(cache_key, data)
            time.sleep(SLEEP)
            return data
        except Exception as e:  # pragma: no cover - network failure
            last_exc = e
            time.sleep(SLEEP)
    raise RuntimeError(f"FBR API failed: {url} params={params} err={last_exc}")


def _flatten_stats(d: Dict[str, Any] | None) -> Dict[str, Any]:
    """Recursively flatten a nested stats dictionary."""

    def _flatten(current: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for k, v in (current or {}).items():
            new_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_flatten(v, new_key + "_"))
            else:
                items[new_key] = v
        return items

    return _flatten(d or {})


def fetch_teams(league_id: int, season_id: str) -> pd.DataFrame:
    data = _get(
        "/teams",
        {"league_id": league_id, "season_id": season_id},
        cache_key=f"teams_{league_id}_{season_id}.json",
    )
    df = pd.DataFrame(data.get("data", []))
    cols = [c for c in ["team", "team_id"] if c in df.columns]
    return df[cols].drop_duplicates()


def fetch_matches(league_id: int, season_id: str) -> pd.DataFrame:
    data = _get(
        "/matches",
        {"league_id": league_id, "season_id": season_id},
        cache_key=f"matches_{league_id}_{season_id}.json",
    )
    df = pd.DataFrame(data.get("data", []))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["season_id"] = season_id
    return df


def fetch_team_match_stats(team_id: int, league_id: int, season_id: str) -> pd.DataFrame:
    data = _get(
        "/team-match-stats",
        {"team_id": team_id, "league_id": league_id, "season_id": season_id},
        cache_key=f"teamstats_{team_id}_{league_id}_{season_id}.json",
    )

    rows = []
    for row in data.get("data", []):
        base = {k: v for k, v in row.items() if k != "stats"}
        stats = row.get("stats") or {}
        base.update(_flatten_stats(stats))
        rows.append(base)

    df = pd.DataFrame(rows)
    core_cols = [
        "match_id",
        "date",
        "team_id",
        "opponent_id",
        "team",
        "gf",
        "ga",
        "xg",
        "xga",
        "shots",
        "sot",
        "poss",
        "home_away",
    ]
    for c in core_cols:
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["season_id"] = season_id
    ordered = core_cols + [c for c in df.columns if c not in core_cols]
    return df[ordered]


def _assemble_one_season(league_id: int, season_id: str) -> pd.DataFrame:
    teams = fetch_teams(league_id, season_id)
    stats = []
    for _, t in teams.iterrows():
        tid = int(t["team_id"])
        stats.append(fetch_team_match_stats(tid, league_id, season_id))
    stats = pd.concat(stats, ignore_index=True)

    home = stats[stats["home_away"].astype(str).str.lower().eq("home")].copy()
    away = stats[stats["home_away"].astype(str).str.lower().eq("away")].copy()

    h = home.rename(columns=lambda c: f"{c}_H" if c not in ["match_id", "date", "season_id"] else c)
    a = away.rename(columns=lambda c: f"{c}_A" if c not in ["match_id", "date", "season_id"] else c)
    Xdf = h.merge(a, on=["match_id", "date", "season_id"], how="inner")

    Xdf["goals_total"] = Xdf["gf_H"].astype(float) + Xdf["gf_A"].astype(float)
    Xdf["over25"] = (Xdf["goals_total"] > 2).astype(int)
    Xdf["outcome"] = np.select(
        [Xdf["gf_H"] > Xdf["gf_A"], Xdf["gf_H"] < Xdf["gf_A"]],
        ["H", "A"],
        default="D",
    )
    return Xdf


def _add_elo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match ELO ratings for home and away teams.

    The calculation uses :func:`calculate_elo_ratings` on all matches up to
    (but not including) the current one so the ratings represent the teams'
    strength entering the fixture.
    """

    # Prepare a minimal DataFrame in the format expected by
    # ``calculate_elo_ratings`` and ensure chronological order.
    matches = (
        df[
            ["date", "team_H", "team_A", "gf_H", "gf_A"]
        ]
        .rename(
            columns={
                "date": "Date",
                "team_H": "HomeTeam",
                "team_A": "AwayTeam",
                "gf_H": "FTHG",
                "gf_A": "FTAG",
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    elo_home: list[float] = []
    elo_away: list[float] = []

    for i in range(len(matches)):
        past = matches.iloc[:i]
        elo_dict = calculate_elo_ratings(past) if i > 0 else {}
        row = matches.iloc[i]
        elo_home.append(float(elo_dict.get(row.HomeTeam, 1500)))
        elo_away.append(float(elo_dict.get(row.AwayTeam, 1500)))

    df = df.sort_values("date").reset_index(drop=True)
    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df


def build_three_seasons(league_id: int, seasons: Iterable[str]) -> pd.DataFrame:
    parts = [_assemble_one_season(league_id, s) for s in seasons]
    df = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)

    def add_roll(df_in: pd.DataFrame, side: str, window: int = 5) -> pd.DataFrame:
        df = df_in.sort_values("date").copy()
        team_col = f"team_id_{side}"
        if team_col not in df.columns:
            df[team_col] = np.nan
        grp = df.groupby(team_col, dropna=False)
        for base in ["xg", "xga", "shots", "sot", "gf", "ga", "poss"]:
            c = f"{base}_{side}"
            if c not in df.columns:
                df[c] = np.nan
            r = grp[c].rolling(window, min_periods=max(3, window // 2)).mean().reset_index(level=0, drop=True)
            df[f"{base}_roll{window}_{side}"] = r.shift(1)
        return df

    for side in ["H", "A"]:
        df = add_roll(df, side, window=5)

    def compute_time_weights(dates: pd.Series, half_life_days: int = 240) -> np.ndarray:
        max_d = pd.to_datetime(dates).max()
        age = (max_d - pd.to_datetime(dates)).dt.days.clip(lower=0).astype(float)
        lam = np.log(2) / max(half_life_days, 1)
        w = np.exp(-lam * age)
        return (w / (w.mean() if w.mean() > 0 else 1.0)).values

    df["sample_weight"] = compute_time_weights(df["date"], half_life_days=240)

    # Add ELO-based strength metrics for each matchup.
    df = _add_elo_columns(df)

    keep = [
        "match_id",
        "date",
        "season_id",
        "team_id_H",
        "team_H",
        "opponent_id_H",
        "team_id_A",
        "team_A",
        "opponent_id_A",
        "xg_H",
        "xga_H",
        "shots_H",
        "sot_H",
        "gf_H",
        "ga_H",
        "poss_H",
        "xg_A",
        "xga_A",
        "shots_A",
        "sot_A",
        "gf_A",
        "ga_A",
        "poss_A",
        "xg_roll5_H",
        "xga_roll5_H",
        "shots_roll5_H",
        "sot_roll5_H",
        "gf_roll5_H",
        "ga_roll5_H",
        "poss_roll5_H",
        "xg_roll5_A",
        "xga_roll5_A",
        "shots_roll5_A",
        "sot_roll5_A",
        "gf_roll5_A",
        "ga_roll5_A",
        "poss_roll5_A",
        "elo_home",
        "elo_away",
        "elo_diff",
        "over25",
        "outcome",
        "sample_weight",
    ]
    return df[keep].copy()


__all__ = [
    "fetch_teams",
    "fetch_matches",
    "fetch_team_match_stats",
    "build_three_seasons",
]

