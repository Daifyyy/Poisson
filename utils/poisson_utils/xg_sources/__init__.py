from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import logging

CACHE_FILE = Path(__file__).with_name("xg_cache.json")

logger = logging.getLogger(__name__)
if not logger.handlers:
    log_dir = Path(__file__).resolve().parents[3] / "logs"
    log_dir.mkdir(exist_ok=True)
    handler = logging.FileHandler(log_dir / "xg_source.log")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
    """Return team xG and xGA for a season from the FBR API.

    Results are cached on disk to avoid repeated network calls.
    """
    cache = _load_cache()
    source = "fbrapi"
    key = f"{season}|{team}|{source}"
    if key in cache:
        data = cache[key]
        if "xg" in data and "xga" in data:
            logger.info(
                "xG source %s for %s %s: xG=%s xGA=%s",
                source,
                season,
                team,
                data["xg"],
                data["xga"],
            )
            return {**data, "source": source}
    try:
        module = import_module(".fbrapi", __name__)
        provider_fn = getattr(module, "get_team_xg_xga")
        data = provider_fn(team, season)
    except Exception:
        data = {}
    if isinstance(data, dict) and "xg" in data and "xga" in data:
        cache[key] = {"xg": data["xg"], "xga": data["xga"]}
        _save_cache(cache)
        logger.info(
            "xG source %s for %s %s: xG=%s xGA=%s",
            source,
            season,
            team,
            data["xg"],
            data["xga"],
        )
        return {"xg": data["xg"], "xga": data["xga"], "source": source}

    logger.warning("No xG data for %s %s", season, team)
    return {"xg": float("nan"), "xga": float("nan"), "source": None}
