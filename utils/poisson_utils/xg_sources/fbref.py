"""Retrieve team expected goals data from FBref.

The real FBref website contains a wealth of statistics.  For the purposes of
this project we only need a tiny subset – the season totals for xG and xGA for a
single team.  The data is cached on disk so repeated calls for the same team do
not hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

CACHE_FILE = Path(__file__).with_name("fbref_xg_cache.json")


def _load_cache() -> Dict[str, Dict[str, float]]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: Dict[str, Dict[str, float]]) -> None:
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def fetch_fbref_team_xg(team: str, season: str, league_code: str) -> Optional[Dict[str, float]]:
    """Return average xG and xGA per game for ``team`` in ``season``.

    The function respects the site's robots.txt and stores results in
    ``CACHE_FILE`` so subsequent calls are served from disk.
    """

    cache = _load_cache()
    key = f"{season}|{league_code}|{team}"
    if key in cache:
        return cache[key]

    url = f"https://fbref.com/en/comps/{league_code}/{season}"

    rp = robotparser.RobotFileParser()
    try:
        rp.set_url("https://fbref.com/robots.txt")
        rp.read()
        if not rp.can_fetch("*", url):
            return None
    except Exception:
        return None

    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return None

    table = soup.find("table", id="stats")
    if table is None:
        return None

    for row in table.find_all("tr"):
        th = row.find("th", {"data-stat": "team"})
        if th is None:
            continue
        name = th.get_text(strip=True)
        if name.lower() != team.lower():
            continue
        try:
            games = float(row.find("td", {"data-stat": "games"}).get_text(strip=True))
            xg_for = float(row.find("td", {"data-stat": "xg_for"}).get_text(strip=True))
            xg_against = float(row.find("td", {"data-stat": "xg_against"}).get_text(strip=True))
        except Exception:
            return None
        games = games or 1.0
        xg = xg_for / games
        xga = xg_against / games
        result = {"xg": xg, "xga": xga}
        cache[key] = result
        _save_cache(cache)
        return result
    return None


def get_team_xg_xga(team: str, season: str, league_code: str) -> Dict[str, float]:
    """Public wrapper used by the provider chain."""
    return fetch_fbref_team_xg(team, season, league_code) or {}


__all__ = ["fetch_fbref_team_xg", "get_team_xg_xga"]
