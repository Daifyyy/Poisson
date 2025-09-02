from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import requests

CACHE_FILE = Path(__file__).with_name("fbrapi_xg_cache.json")


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


def fetch_fbrapi_team_xg(team: str, season: str) -> Optional[Dict[str, float]]:
    """Fetch xG and xGA for a team from the FBR API.

    Returns a dictionary with keys ``xg`` and ``xga`` if available,
    otherwise ``None``.
    """
    api_key = os.getenv("FBRAPI_KEY")
    if not api_key:
        return None

    cache = _load_cache()
    cache_key = f"{season}|{team}"
    if cache_key in cache:
        return cache[cache_key]

    url = "https://fbrapi.com/api/xg"
    params = {"team": team, "season": season, "api_key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    if "xg" in data and "xga" in data:
        result = {"xg": float(data["xg"]), "xga": float(data["xga"])}
        cache[cache_key] = result
        _save_cache(cache)
        return result
    return None


def get_team_xg_xga(team: str, season: str) -> Dict[str, float]:
    """Return team xG and xGA fetched from the FBR API."""
    return fetch_fbrapi_team_xg(team, season) or {}

