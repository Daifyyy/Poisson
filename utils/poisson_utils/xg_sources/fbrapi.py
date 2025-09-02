from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import requests

# On-disk cache for team xG/xGA results
CACHE_FILE = Path(__file__).with_name("fbrapi_xg_cache.json")

# Location where the generated API key is stored so that we do not have to
# hit the ``/generate_api_key`` endpoint on every run. The file lives in the
# user's home directory which is outside of the repository and therefore will
# not be committed.
API_KEY_FILE = Path.home() / ".fbrapi_api_key"


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


def get_fbrapi_api_key() -> Optional[str]:
    """Return an API key for the FBR API, generating and caching it if needed.

    The key is looked up in the ``FBRAPI_KEY`` environment variable first. If
    not present, we try to read it from ``API_KEY_FILE``. As a last resort the
    key is requested from ``https://fbrapi.com/generate_api_key`` and stored to
    the cache file for subsequent runs.
    """

    env_key = os.getenv("FBRAPI_KEY")
    if env_key:
        return env_key.strip()

    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            os.environ["FBRAPI_KEY"] = key
            return key

    try:
        resp = requests.post("https://fbrapi.com/generate_api_key", timeout=10)
        resp.raise_for_status()
        key = resp.json().get("api_key")
    except Exception:
        return None

    if key:
        API_KEY_FILE.write_text(key, encoding="utf-8")
        os.environ["FBRAPI_KEY"] = key
        return key
    return None


def fetch_fbrapi_team_xg(team: str, season: str) -> Optional[Dict[str, float]]:
    """Fetch xG and xGA for a team from the FBR API.

    Returns a dictionary with keys ``xg`` and ``xga`` if available,
    otherwise ``None``. The function automatically handles API key
    generation and caching so the key request is made only once.
    """
    api_key = get_fbrapi_api_key()
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

