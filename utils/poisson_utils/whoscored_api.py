"""Utility for fetching and caching team-level xG data from WhoScored.

The module provides a single public function ``get_whoscored_xg`` which
retrieves the expected goals (xG) value for a given team.  Results are cached on
 disk to avoid hitting the remote service repeatedly.  Basic retry logic with
exponential backoff is implemented to gracefully handle rate limits.

The WhoScored API is not publicly documented; this utility expects an endpoint
that returns JSON data containing an ``xG`` field.  If the endpoint changes or
 returns unexpected data, the function will simply return ``float('nan')``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import requests

# Location of the on-disk cache.  It lives next to this module so it persists
# between runs but remains inside the repository tree.
CACHE_FILE = Path(__file__).with_name("whoscored_xg_cache.json")

# Template URL for fetching data from WhoScored.  The ``team`` placeholder is
# substituted with the provided team name.  This may need adjustment if
# WhoScored changes their API.
API_URL = "https://www.whoscored.com/api/team/{team}/xg"

# In-memory cache loaded from disk.  Keys are lower-case team names.
_cache: Dict[str, float] = {}


def _load_cache() -> None:
    """Populate the in-memory cache from ``CACHE_FILE`` if it exists."""
    global _cache
    if CACHE_FILE.exists():
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _cache = {}
    else:
        _cache = {}


def _save_cache() -> None:
    """Persist the in-memory cache to ``CACHE_FILE``."""
    try:
        CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
    except OSError:
        # Failing to save the cache is non-critical; ignore the error.
        pass


# Load cache immediately so lookups are fast and do not trigger file I/O on
# repeated calls.
_load_cache()


def _fetch_xg_from_api(team_name: str) -> float:
    """Fetch xG for ``team_name`` from the remote WhoScored API.

    If the API responds with HTTP 429 (rate limit) the request is retried up to
    three times with exponential backoff.  Any network error or missing data
    results in ``float('nan')`` being returned.
    """
    url = API_URL.format(team=team_name)
    backoff = 1
    for _ in range(3):
        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException:
            return float("nan")

        if response.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            continue

        if response.status_code != 200:
            return float("nan")

        try:
            data = response.json()
        except ValueError:
            return float("nan")

        xg = data.get("xG")
        return float(xg) if xg is not None else float("nan")

    return float("nan")


def get_whoscored_xg(team_name: str) -> float:
    """Return the xG value for ``team_name`` fetched from WhoScored.

    The result is cached to reduce network requests.  If the data is unavailable
    or an error occurs, ``float('nan')`` is returned.
    """
    key = team_name.lower()
    if key in _cache:
        return _cache[key]

    xg = _fetch_xg_from_api(team_name)
    _cache[key] = xg
    _save_cache()
    return xg

