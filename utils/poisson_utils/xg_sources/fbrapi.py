"""Retrieve team expected goals data from FBref via FBRAPI.

Updated to use correct API endpoints and authentication method.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# On-disk cache for team xG/xGA results
CACHE_FILE = Path(__file__).with_name("fbrapi_xg_cache.json")

# Directory for storing raw FBR API responses for debugging/inspection
RAW_OUTPUT_DIR = Path(__file__).with_name("fbrapi_raw")
RAW_OUTPUT_DIR.mkdir(exist_ok=True)


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


def _save_raw_response(filename: str, data: Dict[str, Any]) -> None:
    """Persist raw API responses for later inspection."""
    try:
        with (RAW_OUTPUT_DIR / filename).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_fbrapi_api_key() -> Optional[str]:
    """Return an API key for the FBR API.

    The key is read from the ``FBRAPI`` environment variable.
    """

    env_key = os.getenv("FBRAPI")
    if env_key:
        return env_key.strip()

    return None


def find_team_in_standings(team_name: str, league_id: int, season_id: str, api_key: str) -> Optional[Dict[str, float]]:
    """Find team xG/xGA data from league standings."""
    
    url = "https://fbrapi.com/league-standings"
    headers = {"X-API-Key": api_key}
    params = {"league_id": league_id, "season_id": season_id}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"Standings request status: {resp.status_code}")

        if resp.status_code != 200:
            print(f"Standings error response: {resp.text}")
            return None

        data = resp.json()
        _save_raw_response(f"standings_{league_id}_{season_id}.json", data)
        
        # Hledáme tým ve všech tabulkách standings
        for standings_table in data.get("data", []):
            for team_data in standings_table.get("standings", []):
                if team_data.get("team_name", "").lower() == team_name.lower():
                    # Počítáme průměrné xG a xGA na zápas
                    mp = team_data.get("mp", 1) or 1  # Matches played
                    xg_total = team_data.get("xg", 0)
                    xga_total = team_data.get("xga", 0)
                    
                    return {
                        "xg": float(xg_total) / float(mp),
                        "xga": float(xga_total) / float(mp)
                    }
        
        return None
        
    except Exception as e:
        print(f"Error fetching standings: {e}")
        return None


def find_team_in_season_stats(team_name: str, league_id: int, season_id: str, api_key: str) -> Optional[Dict[str, float]]:
    """Find team xG/xGA data from team season stats."""
    
    url = "https://fbrapi.com/team-season-stats"
    headers = {"X-API-Key": api_key}
    params = {"league_id": league_id, "season_id": season_id}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"Season stats request status: {resp.status_code}")

        if resp.status_code != 200:
            print(f"Season stats error response: {resp.text}")
            return None

        data = resp.json()
        _save_raw_response(f"season_stats_{league_id}_{season_id}.json", data)
        
        # Hledáme tým v datech
        for team_data in data.get("data", []):
            team_meta = team_data.get("meta_data", {})
            if team_meta.get("team_name", "").lower() == team_name.lower():
                stats = team_data.get("stats", {})
                
                # Různé ligy mohou mít různé názvy polí
                # Zkusíme najít xG a xGA data v různých formátech
                mp = stats.get("mp", stats.get("games_played", 1)) or 1
                
                # Možné názvy pro xG data
                xg_total = (stats.get("xg") or stats.get("xg_for") or 
                           stats.get("expected_goals") or 0)
                xga_total = (stats.get("xga") or stats.get("xg_against") or 
                            stats.get("expected_goals_against") or 0)
                
                if xg_total or xga_total:
                    return {
                        "xg": float(xg_total) / float(mp),
                        "xga": float(xga_total) / float(mp)
                    }
        
        return None
        
    except Exception as e:
        print(f"Error fetching team season stats: {e}")
        return None


def fetch_fbrapi_team_xg(team: str, season: str, league_id: int = None) -> Optional[Dict[str, float]]:
    """Fetch xG and xGA for a team from the FBR API.

    Args:
        team: Team name
        season: Season ID (e.g., "2023-24" or "2024")
        league_id: League ID (if known, helps narrow search)

    Returns:
        Dictionary with keys 'xg' and 'xga' if available, otherwise None.
    """
    api_key = get_fbrapi_api_key()
    if not api_key:
        print("Could not obtain API key")
        return None

    cache = _load_cache()
    cache_key = f"{season}|{league_id}|{team}"
    if cache_key in cache:
        return cache[cache_key]

    result = None
    
    # Pokud známe league_id, zkusíme nejdřív season stats
    if league_id:
        result = find_team_in_season_stats(team, league_id, season, api_key)
        if not result:
            result = find_team_in_standings(team, league_id, season, api_key)
    else:
        # Pokud neznáme league_id, musíme najít všechny ligy a hledat
        print("No league_id provided - you should specify league_id for better performance")
        # Tento případ by vyžadoval iteraci přes všechny dostupné ligy
        # což je mimo rámec tohoto příkladu
    
    if result:
        cache[cache_key] = result
        _save_cache(cache)
        
    return result


def get_team_xg_xga(team: str, season: str, league_id: int = None) -> Dict[str, float]:
    """Public wrapper used by the provider chain.
    
    Args:
        team: Team name
        season: Season ID
        league_id: League ID (recommended to specify)
    """
    result = fetch_fbrapi_team_xg(team, season, league_id)
    return result or {}


def test_api_connection() -> bool:
    """Test if API key works by trying to get countries data."""
    api_key = get_fbrapi_api_key()
    if not api_key:
        return False
        
    url = "https://fbrapi.com/countries"
    headers = {"X-API-Key": api_key}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"Test connection status: {resp.status_code}")
        if resp.status_code == 200:
            print("API connection successful!")
            return True
        else:
            print(f"API test failed: {resp.text}")
            return False
    except Exception as e:
        print(f"Connection test error: {e}")
        return False


def get_available_leagues(country_code: str = None) -> Optional[Dict]:
    """Get available leagues, optionally filtered by country."""
    api_key = get_fbrapi_api_key()
    if not api_key:
        return None
        
    headers = {"X-API-Key": api_key}
    
    if country_code:
        url = "https://fbrapi.com/leagues"
        params = {"country_code": country_code}
    else:
        url = "https://fbrapi.com/countries"
        params = {}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Error getting leagues: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


__all__ = ["fetch_fbrapi_team_xg", "get_team_xg_xga", "test_api_connection", "get_available_leagues"]
