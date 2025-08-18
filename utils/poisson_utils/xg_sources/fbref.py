from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup
from urllib import robotparser

CACHE_FILE = Path(__file__).with_name("fbref_xg_cache.json")


def _load_cache() -> Dict[str, Dict[str, float]]:
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: Dict[str, Dict[str, float]]) -> None:
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f)


def _parse_float(text: str) -> float:
    return float(text.replace(",", "").strip())


def fetch_fbref_team_xg(team: str, season: str, league: str) -> Optional[Dict[str, float]]:
    """Fetch average xG and xGA per match for a team from FBRef.

    Parameters
    ----------
    team: str
        Team name as displayed on FBRef.
    season: str
        Season identifier, e.g. ``"2023-2024"``.
    league: str
        League identifier used by FBRef (e.g. ``"9"`` for Premier League).

    Returns
    -------
    Optional[Dict[str, float]]
        Dictionary with keys ``"xg"`` and ``"xga"`` containing per-match
        expected goals for and against. ``None`` is returned on network
        failures or if the page cannot be retrieved. Parsing problems raise
        ``ValueError``.
    """
    cache = _load_cache()
    key = f"{season}|{league}|{team}"
    if key in cache:
        return cache[key]

    path = f"/en/comps/{league}/{season}/stats"
    url = f"https://fbref.com{path}"

    rp = robotparser.RobotFileParser()
    rp.set_url("https://fbref.com/robots.txt")
    try:
        rp.read()
    except Exception:
        return None
    if not rp.can_fetch("*", path):
        raise RuntimeError("Fetching disallowed by robots.txt")

    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    row = None
    for th in soup.find_all("th", {"data-stat": "team"}):
        if th.get_text(strip=True) == team:
            row = th.parent
            break
    if row is None:
        raise ValueError("team row not found")

    try:
        games = _parse_float(row.find("td", {"data-stat": "games"}).get_text())
        xg_total = _parse_float(row.find("td", {"data-stat": "xg_for"}).get_text())
        xga_total = _parse_float(row.find("td", {"data-stat": "xg_against"}).get_text())
    except Exception as exc:
        raise ValueError("required columns missing") from exc
    if games == 0:
        raise ValueError("games value is zero")

    result = {"xg": xg_total / games, "xga": xga_total / games}
    cache[key] = result
    _save_cache(cache)
    return result


def get_team_xg_xga(team: str, season: str, league: Optional[str] = None) -> Dict[str, float]:
    if league is None:
        return {}
    return fetch_fbref_team_xg(team, season, league) or {}
