# fbrapi_dataset.py
from __future__ import annotations

import time
import math
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests


BASE = "https://fbrapi.com"
API_KEY = "PASTE_YOUR_API_KEY_HERE"   # <<< DOPLŇ
HDRS = {"X-API-Key": API_KEY}
SLEEP = 3.2          # FBR API uvádí ~1 dotaz / 3 s
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
        c = _cached_json(cache_key)
        if c is not None:
            return c

    url = f"{BASE}{path}"
    last_exc = None
    for _ in range(RETRIES):
        try:
            r = requests.get(url, params=params or {}, headers=HDRS, timeout=60)
            if r.status_code == 429:
                # rate limit – počkej a zkus znovu
                time.sleep(max(SLEEP, 3.0))
                continue
            r.raise_for_status()
            data = r.json()
            if cache_key:
                _save_cache(cache_key, data)
            time.sleep(SLEEP)  # respektuj limit
            return data
        except Exception as e:
            last_exc = e
            time.sleep(SLEEP)
    raise RuntimeError(f"FBR API request failed: {url} params={params} err={last_exc}")


def fetch_teams(league_id: int, season_id: str) -> pd.DataFrame:
    data = _get("/teams", {"league_id": league_id, "season_id": season_id},
                cache_key=f"teams_{league_id}_{season_id}.json")
    df = pd.DataFrame(data.get("data", []))
    # očekáváme: team, team_id, atd.
    cols = [c for c in ["team", "team_id"] if c in df.columns]
    return df[cols].drop_duplicates()


def fetch_matches(league_id: int, season_id: str) -> pd.DataFrame:
    data = _get("/matches", {"league_id": league_id, "season_id": season_id},
                cache_key=f"matches_{league_id}_{season_id}.json")
    df = pd.DataFrame(data.get("data", []))
    # typicky: match_id, date, home_away, team_id, opponent_id, gf, ga, result, ...
    # Ujisti datum:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def fetch_team_match_stats(team_id: int, league_id: int, season_id: str) -> pd.DataFrame:
    data = _get("/team-match-stats",
                {"team_id": team_id, "league_id": league_id, "season_id": season_id},
                cache_key=f"teamstats_{team_id}_{league_id}_{season_id}.json")
    df = pd.DataFrame(data.get("data", []))
    # očekávané klíče – doplníme když chybí
    for c in ["match_id","team_id","opponent_id","team","gf","ga","xg","xga","shots","sot","poss"]:
        if c not in df.columns:
            df[c] = np.nan
    # Datum:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df[["match_id","date","team_id","opponent_id","team","gf","ga","xg","xga","shots","sot","poss"]]


def build_match_frame(league_id: int, season_id: str) -> pd.DataFrame:
    teams = fetch_teams(league_id, season_id)
    matches = fetch_matches(league_id, season_id)

    # Stáhni per-team stats a spoj
    stats_list: List[pd.DataFrame] = []
    for _, t in teams.iterrows():
        tid = int(t["team_id"])
        sdf = fetch_team_match_stats(tid, league_id, season_id)
        stats_list.append(sdf)
    stats = pd.concat(stats_list, ignore_index=True)

    # Očekáváme, že každý match_id má dva řádky ve stats (HOME/AWAY).
    # V matches obvykle existuje sloupec "home_away" vztahovaný k team_id.
    # Přilepíme "home_away" do stats podle (match_id, team_id) když to jde;
    # pokud ne, určujeme HOME/AWAY podle gf vs ga z pohledu obou stran (fallback).
    key_cols = [c for c in ["match_id", "team_id", "home_away"] if c in matches.columns]
    if set(key_cols) == {"match_id", "team_id", "home_away"}:
        m = matches[key_cols].drop_duplicates()
        stats = stats.merge(m, on=["match_id","team_id"], how="left")
    else:
        stats["home_away"] = np.nan  # fallback – určíme níž

    # Rozděl na home a away
    home = stats[stats["home_away"].astype(str).str.lower().eq("home")].copy()
    away = stats[stats["home_away"].astype(str).str.lower().eq("away")].copy()

    # Fallback: pokud je home/away prázdné, zkusíme doplnit tak, aby match_id měl přesně 2 řádky,
    # a „home“ bude ten, který má totéž match_id jako druhý řádek – bez spolehlivého příznaku to jde hůř.
    # Poslední záchrana: určeme HOME/AWAY podle toho, kde gf - ga koreluje s tím, co je v matches „gf_home“ atp.
    # (Abychom to zjednodušili: pokud po prvním kroku něco chybí, složíme perechet z dvojic na shodné match_id.)
    if home.empty or away.empty:
        # spáruj po dvojicích stejný match_id: první jako home, druhý jako away (deterministicky)
        tmp = stats.sort_values(["match_id","team_id"]).groupby("match_id").head(2)
        h = tmp.groupby("match_id").nth(0).reset_index().rename(columns=lambda c: f"{c}_H" if c != "match_id" else c)
        a = tmp.groupby("match_id").nth(1).reset_index().rename(columns=lambda c: f"{c}_A" if c != "match_id" else c)
        Xdf = h.merge(a, on="match_id", how="inner")
    else:
        h = home.rename(columns=lambda c: f"{c}_H" if c not in ["match_id","date"] else c)
        a = away.rename(columns=lambda c: f"{c}_A" if c not in ["match_id","date"] else c)
        # sjednoť datum (měl by být stejný)
        if "date" in h.columns and "date" in a.columns:
            Xdf = h.merge(a, on="match_id", suffixes=("", ""), how="inner")
            # preferuj date z home:
            if "date_x" in Xdf.columns and "date_y" in Xdf.columns:
                Xdf["date"] = Xdf["date_x"].fillna(Xdf["date_y"])
                Xdf.drop(columns=["date_x","date_y"], inplace=True)
        else:
            Xdf = h.merge(a, on="match_id", how="inner")

    # Ujisti datum:
    if "date" in Xdf.columns:
        Xdf["date"] = pd.to_datetime(Xdf["date"], errors="coerce")

    # Cílovka (Over2.5) – součet gólů v zápase:
    Xdf["goals_total"] = (Xdf.get("gf_H", np.nan)).astype(float) + (Xdf.get("gf_A", np.nan)).astype(float)
    Xdf["over25"] = (Xdf["goals_total"] > 2).astype(int)

    # Rolling featury bez leakage
    def add_rolling_features(df: pd.DataFrame, side: str, window: int = 5) -> pd.DataFrame:
        df = df.sort_values("date")
        # group podle team_id_H / _A
        team_col = f"team_id_{side}"
        df[team_col] = df[team_col].astype(float)
        grp = df.groupby(team_col, dropna=False)

        for base in ["xg", "xga", "shots", "sot", "gf", "ga", "poss"]:
            col = f"{base}_{side}"
            if col not in df.columns:
                df[col] = np.nan
            roll = grp[col].rolling(window, min_periods=max(3, window//2)).mean().reset_index(level=0, drop=True)
            df[f"{base}_roll{window}_{side}"] = roll.shift(1)  # shift => pouze minulost
        return df

    for side in ["H", "A"]:
        Xdf = add_rolling_features(Xdf, side)

    # Vyber, co budeme exportovat dál (featury + cílovka + identifikace)
    keep = [
        "match_id", "date", "over25",
        "team_id_H","team_id_A","opponent_id_H","opponent_id_A","team_H","team_A",
        # raw
        "xg_H","xga_H","shots_H","sot_H","gf_H","ga_H","poss_H",
        "xg_A","xga_A","shots_A","sot_A","gf_A","ga_A","poss_A",
        # rolling
        "xg_roll5_H","xga_roll5_H","shots_roll5_H","sot_roll5_H","gf_roll5_H","ga_roll5_H","poss_roll5_H",
        "xg_roll5_A","xga_roll5_A","shots_roll5_A","sot_roll5_A","gf_roll5_A","ga_roll5_A","poss_roll5_A",
    ]
    # některé nemusí existovat (fallback cesta); filtruj dle přítomnosti
    keep = [c for c in keep if c in Xdf.columns]
    return Xdf[keep].dropna(subset=["over25"]).reset_index(drop=True)
