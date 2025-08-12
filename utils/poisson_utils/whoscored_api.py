"""Utility for fetching and caching team-level xG/xGA data from WhoScored.

The module exposes two public helpers:

``get_whoscored_xg``
    Returns only the expected goals (xG) value for backwards compatibility.

``get_whoscored_xg_xga``
    Returns a dictionary with both ``xg`` and ``xga`` for a given team.  The
    results are cached on disk to avoid hitting the remote service repeatedly
    and basic retry logic with exponential backoff is implemented to gracefully
    handle rate limits.

The WhoScored API is not publicly documented; this utility expects an endpoint
that returns JSON data containing ``xG`` and ``xGA`` fields.  If the endpoint
changes or returns unexpected data, ``float('nan')`` values are used.
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

# In-memory cache loaded from disk.  Keys are lower-case team names and map to
# dictionaries with ``xg`` and ``xga`` values.
_cache: Dict[str, Dict[str, float]] = {}


def _load_cache() -> None:
    """Populate the in-memory cache from ``CACHE_FILE`` if it exists."""
    global _cache
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}

        # Older versions stored just floats.  Convert to the new structure.
        if isinstance(data, dict):
            converted = {}
            for key, val in data.items():
                if isinstance(val, dict):
                    converted[key] = {
                        "xg": float(val.get("xg", float("nan"))),
                        "xga": float(val.get("xga", float("nan"))),
                    }
                else:  # old format ``team -> xg``
                    converted[key] = {"xg": float(val), "xga": float("nan")}
            _cache = converted
        else:
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


def _fetch_xg_xga_from_api(team_name: str) -> Dict[str, float]:
    """Fetch xG and xGA for ``team_name`` from the remote WhoScored API.

    If the API responds with HTTP 429 (rate limit) the request is retried up to
    three times with exponential backoff.  Any network error or missing data
    results in ``float('nan')`` values being returned.
    """
    url = API_URL.format(team=team_name)
    backoff = 1
    for _ in range(3):
        try:
            response = requests.get(url, timeout=10)
        except requests.RequestException:
            return {"xg": float("nan"), "xga": float("nan")}

        if response.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            continue

        if response.status_code != 200:
            return {"xg": float("nan"), "xga": float("nan")}

        try:
            data = response.json()
        except ValueError:
            return {"xg": float("nan"), "xga": float("nan")}

        xg = data.get("xG") or data.get("xg")
        xga = data.get("xGA") or data.get("xga")
        return {
            "xg": float(xg) if xg is not None else float("nan"),
            "xga": float(xga) if xga is not None else float("nan"),
        }

    return {"xg": float("nan"), "xga": float("nan")}


def get_whoscored_xg_xga(team_name: str) -> Dict[str, float]:
    """Return a dictionary with ``xg`` and ``xga`` for ``team_name``.

    The result is cached to avoid repeated network calls.  Missing values are
    represented as ``float('nan')``.
    """
    key = team_name.lower()
    if key in _cache:
        return _cache[key]

    stats = _fetch_xg_xga_from_api(team_name)
    _cache[key] = stats
    _save_cache()
    return stats


def get_whoscored_xg(team_name: str) -> float:
    """Return only the xG value for ``team_name``.

    This is a thin wrapper around :func:`get_whoscored_xg_xga` kept for
    backwards compatibility with existing code that expects a single float.
    """
    stats = get_whoscored_xg_xga(team_name)
    return stats.get("xg", float("nan"))

