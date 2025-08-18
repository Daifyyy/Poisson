from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import requests

CACHE_FILE = Path(__file__).with_name("understat_xg_cache.json")


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


def fetch_understat_team_xg(team: str, season: str) -> Optional[Dict[str, float]]:
    """Fetch average xG and xGA per match for a team and season.

    Returns ``None`` if the page cannot be retrieved, rate limited or the schema
    is not as expected.
    """
    cache = _load_cache()
    key = f"{season}|{team}"
    if key in cache:
        return cache[key]

    url = f"https://understat.com/team/{team}/{season}"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None

    html = resp.text
    match = re.search(r"shotsData\s*=\s*JSON.parse\('([^']+)'\)", html)
    if not match:
        return None

    raw_json = match.group(1)
    try:
        decoded = raw_json.encode("utf-8").decode("unicode_escape")
        shots_data = json.loads(decoded)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    total_xg = 0.0
    total_xga = 0.0
    matches = 0

    try:
        items = shots_data.values() if isinstance(shots_data, dict) else shots_data
        for entry in items:
            matches += 1
            for shot in entry.get("h", []):
                xg = float(shot.get("xG", 0) or 0)
                if shot.get("h_team") == team:
                    total_xg += xg
                else:
                    total_xga += xg
            for shot in entry.get("a", []):
                xg = float(shot.get("xG", 0) or 0)
                if shot.get("a_team") == team:
                    total_xg += xg
                else:
                    total_xga += xg
    except Exception:
        return None

    if matches == 0:
        return None

    result = {"xg": total_xg / matches, "xga": total_xga / matches}
    cache[key] = result
    _save_cache(cache)
    return result


def get_team_xg_xga(team: str, season: str) -> Dict[str, float]:
    return fetch_understat_team_xg(team, season) or {}
