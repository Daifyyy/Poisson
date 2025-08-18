from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

CACHE_FILE = Path(__file__).with_name("xg_cache.json")


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


def get_team_xg_xga(
    team: str, season: str, league_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Return team xG and xGA for a season from available providers.

    Providers are queried in order: ``understat`` and ``fbref``.  If both
    fail and ``league_df`` is provided, pseudo-xG values computed from the
    dataframe are used. Results are cached on disk to avoid repeated network
    calls.
    """
    cache = _load_cache()
    for source in ("understat", "fbref"):
        key = f"{season}|{team}|{source}"
        if key in cache:
            data = cache[key]
            if "xg" in data and "xga" in data:
                return {**data, "source": source}
        try:
            module = import_module(f".{source}", __name__)
            provider_fn = getattr(module, "get_team_xg_xga")
            data = provider_fn(team, season)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "xg" in data and "xga" in data:
            cache[key] = {"xg": data["xg"], "xga": data["xga"]}
            _save_cache(cache)
            return {"xg": data["xg"], "xga": data["xga"], "source": source}

    if league_df is not None:
        source = "pseudo"
        key = f"{season}|{team}|{source}"
        if key in cache:
            data = cache[key]
            if "xg" in data and "xga" in data:
                return {**data, "source": source}
        try:
            from .pseudo import fetch_pseudo_xg

            data = fetch_pseudo_xg(team, league_df)
        except Exception:
            data = {}
        if isinstance(data, dict) and "xg" in data and "xga" in data:
            cache[key] = {"xg": data["xg"], "xga": data["xga"]}
            _save_cache(cache)
            return {"xg": data["xg"], "xga": data["xga"], "source": source}

    return {"xg": float("nan"), "xga": float("nan"), "source": None}
