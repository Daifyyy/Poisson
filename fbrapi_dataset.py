"""Utilities for downloading match data from the FBR API.

Fixed to work with the actual FBR API endpoints and data structure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

# Assuming these imports exist in your project
try:
    from utils.poisson_utils.xg_sources.fbrapi import get_fbrapi_api_key
    from utils.poisson_utils.elo import calculate_elo_ratings
except ImportError:
    # Fallback if modules don't exist
    def get_fbrapi_api_key():
        return None
    def calculate_elo_ratings(df):
        return {}

BASE = "https://fbrapi.com"
SLEEP = 3.3  # ~1 request / 3 s
RETRIES = 3
CACHE_DIR = Path("cache_fbrapi")
CACHE_DIR.mkdir(exist_ok=True)


def _cached_json(filename: str) -> Dict[str, Any] | None:
    p = CACHE_DIR / filename
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(filename: str, data: Dict[str, Any]) -> None:
    (CACHE_DIR / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_api_key() -> str:
    """Get API key with better error handling."""
    api_key = get_fbrapi_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing FBR API key. Please set the FBRAPI environment variable "
            "or ensure the API key generation is working."
        )
    return api_key


def _get(path: str, params: Dict[str, Any] | None = None, cache_key: str | None = None) -> Dict[str, Any]:
    """Make API request with proper error handling."""
    if cache_key:
        cached = _cached_json(cache_key)
        if cached is not None:
            print(f"Cache hit: {cache_key}")
            return cached

    api_key = _get_api_key()
    headers = {"X-API-Key": api_key}
    url = f"{BASE}{path}"
    
    print(f"API request: {url} with params: {params}")
    
    last_exc: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params or {}, headers=headers, timeout=60)
            print(f"Response status: {r.status_code}")
            
            if r.status_code == 429:
                print("Rate limit hit, waiting...")
                time.sleep(max(SLEEP * 2, 6.0))
                continue
                
            if r.status_code == 401:
                raise RuntimeError("API key is invalid or expired")
                
            if r.status_code == 404:
                print(f"Resource not found: {url}")
                return {"data": []}
                
            r.raise_for_status()
            data = r.json()
            
            if cache_key:
                _save_cache(cache_key, data)
                
            time.sleep(SLEEP)
            return data
            
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"Request failed (attempt {attempt + 1}/{RETRIES}): {e}")
            if attempt < RETRIES - 1:
                time.sleep(SLEEP * (attempt + 1))
                
    raise RuntimeError(f"FBR API failed after {RETRIES} attempts: {url} params={params} err={last_exc}")


def get_league_teams(league_id: int, season_id: str) -> Dict[str, str]:
    """Get mapping of team names to team IDs for a league/season."""
    
    # Try to get teams from league standings
    try:
        data = _get(
            "/league-standings",
            {"league_id": league_id, "season_id": season_id},
            cache_key=f"standings_{league_id}_{season_id}.json",
        )
        
        team_mapping = {}
        for standings_table in data.get("data", []):
            for team_data in standings_table.get("standings", []):
                team_name = team_data.get("team_name")
                team_id = team_data.get("team_id")
                if team_name and team_id:
                    team_mapping[team_name] = team_id
                    
        print(f"Found {len(team_mapping)} teams in league {league_id}")
        return team_mapping
        
    except Exception as e:
        print(f"Could not get teams from standings: {e}")
        return {}


def fetch_teams(league_id: int, season_id: str) -> pd.DataFrame:
    """Fetch teams for a league/season."""
    team_mapping = get_league_teams(league_id, season_id)
    
    if not team_mapping:
        print("No teams found")
        return pd.DataFrame(columns=["team", "team_id"])
    
    teams_data = [{"team": name, "team_id": tid} for name, tid in team_mapping.items()]
    return pd.DataFrame(teams_data)


def fetch_matches(league_id: int, season_id: str) -> pd.DataFrame:
    """Fetch matches for a league/season."""
    data = _get(
        "/matches",
        {"league_id": league_id, "season_id": season_id},
        cache_key=f"matches_{league_id}_{season_id}.json",
    )
    
    matches = data.get("data", [])
    if not matches:
        print("No matches found")
        return pd.DataFrame()
    
    df = pd.DataFrame(matches)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    
    df["season_id"] = season_id
    return df


def _flatten_stats(d: Dict[str, Any] | None) -> Dict[str, Any]:
    """Recursively flatten a nested stats dictionary."""
    def _flatten(current: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for k, v in (current or {}).items():
            new_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_flatten(v, new_key + "_"))
            else:
                items[new_key] = v
        return items

    return _flatten(d or {})


def fetch_team_match_stats(team_id: str, league_id: int, season_id: str) -> pd.DataFrame:
    """Fetch match-level statistics for a specific team."""
    
    data = _get(
        "/team-match-stats",
        {"team_id": team_id, "league_id": league_id, "season_id": season_id},
        cache_key=f"teamstats_{team_id}_{league_id}_{season_id}.json",
    )

    matches = data.get("data", [])
    if not matches:
        print(f"No match stats found for team {team_id}")
        return pd.DataFrame()

    rows = []
    for match in matches:
        meta_data = match.get("meta_data", {})
        stats = match.get("stats", {})
        
        # Combine meta data and flattened stats
        row = {**meta_data, **_flatten_stats(stats)}
        rows.append(row)

    df = pd.DataFrame(rows)
    
    if df.empty:
        return df

    # Standardize column names and ensure required columns exist
    column_mapping = {
        # Map various possible column names to standard ones
        "goals_for": "gf",
        "goals_against": "ga", 
        "expected_goals": "xg",
        "expected_goals_against": "xga",
        "shots_total": "shots",
        "shots_on_target": "sot",
        "possession": "poss",
    }
    
    # Apply column mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Ensure core columns exist
    core_cols = [
        "match_id", "date", "team_id", "opponent_id", "team_name", "opponent",
        "gf", "ga", "xg", "xga", "shots", "sot", "poss", "home_away"
    ]
    
    for col in core_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Convert date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

    df["season_id"] = season_id
    
    # Reorder columns
    ordered_cols = core_cols + ["season_id"] + [c for c in df.columns if c not in core_cols + ["season_id"]]
    available_cols = [c for c in ordered_cols if c in df.columns]
    
    return df[available_cols]


def _assemble_one_season(league_id: int, season_id: str) -> pd.DataFrame:
    """Assemble all match data for one season."""
    
    print(f"Assembling season {season_id} for league {league_id}")
    
    # Get team mapping
    team_mapping = get_league_teams(league_id, season_id)
    if not team_mapping:
        raise RuntimeError(f"No teams found for league {league_id}, season {season_id}")

    # Fetch stats for each team
    all_stats = []
    for team_name, team_id in team_mapping.items():
        print(f"Fetching stats for {team_name} (ID: {team_id})")
        try:
            team_stats = fetch_team_match_stats(team_id, league_id, season_id)
            if not team_stats.empty:
                team_stats["team_name"] = team_name
                all_stats.append(team_stats)
        except Exception as e:
            print(f"Failed to fetch stats for {team_name}: {e}")
            continue

    if not all_stats:
        raise RuntimeError("No team statistics could be fetched")

    # Combine all team stats
    combined_stats = pd.concat(all_stats, ignore_index=True)
    
    # Split into home and away matches
    home = combined_stats[combined_stats["home_away"].astype(str).str.lower() == "home"].copy()
    away = combined_stats[combined_stats["home_away"].astype(str).str.lower() == "away"].copy()

    if home.empty or away.empty:
        raise RuntimeError("Could not separate home and away matches")

    # Rename columns for merge
    home_cols = {}
    away_cols = {}
    
    for col in home.columns:
        if col not in ["match_id", "date", "season_id"]:
            home_cols[col] = f"{col}_H"
            
    for col in away.columns:
        if col not in ["match_id", "date", "season_id"]:
            away_cols[col] = f"{col}_A"

    h = home.rename(columns=home_cols)
    a = away.rename(columns=away_cols)

    # Merge home and away data
    merged = h.merge(a, on=["match_id", "date", "season_id"], how="inner")
    
    if merged.empty:
        raise RuntimeError("No matches could be merged")

    # Calculate derived features
    merged["goals_total"] = pd.to_numeric(merged.get("gf_H", 0), errors="coerce") + pd.to_numeric(merged.get("gf_A", 0), errors="coerce")
    merged["over25"] = (merged["goals_total"] > 2.5).astype(int)
    
    # Calculate match outcome
    gf_h = pd.to_numeric(merged.get("gf_H", 0), errors="coerce").fillna(0)
    gf_a = pd.to_numeric(merged.get("gf_A", 0), errors="coerce").fillna(0)
    
    merged["outcome"] = np.select(
        [gf_h > gf_a, gf_h < gf_a],
        ["H", "A"],
        default="D",
    )
    
    print(f"Successfully assembled {len(merged)} matches for season {season_id}")
    return merged


def _add_elo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match ELO ratings for home and away teams."""
    
    if df.empty:
        return df
    
    # Prepare DataFrame for ELO calculation
    required_cols = ["date", "team_name_H", "team_name_A", "gf_H", "gf_A"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns for ELO calculation: {missing_cols}")
        # Add default ELO columns
        df["elo_home"] = 1500.0
        df["elo_away"] = 1500.0
        df["elo_diff"] = 0.0
        return df

    try:
        matches = (
            df[required_cols]
            .rename(columns={
                "date": "Date",
                "team_name_H": "HomeTeam",
                "team_name_A": "AwayTeam", 
                "gf_H": "FTHG",
                "gf_A": "FTAG",
            })
            .sort_values("Date")
            .reset_index(drop=True)
        )

        elo_home: list[float] = []
        elo_away: list[float] = []

        for i in range(len(matches)):
            past = matches.iloc[:i]
            elo_dict = calculate_elo_ratings(past) if i > 0 else {}
            row = matches.iloc[i]
            elo_home.append(float(elo_dict.get(row.HomeTeam, 1500)))
            elo_away.append(float(elo_dict.get(row.AwayTeam, 1500)))

        df = df.sort_values("date").reset_index(drop=True)
        df["elo_home"] = elo_home
        df["elo_away"] = elo_away
        df["elo_diff"] = df["elo_home"] - df["elo_away"]
        
        print("Successfully calculated ELO ratings")
        
    except Exception as e:
        print(f"Error calculating ELO ratings: {e}")
        df["elo_home"] = 1500.0
        df["elo_away"] = 1500.0  
        df["elo_diff"] = 0.0
        
    return df


def get_league_teams(league_id: int, season_id: str) -> Dict[str, str]:
    """Get mapping of team names to team IDs."""
    
    try:
        data = _get(
            "/league-standings",
            {"league_id": league_id, "season_id": season_id},
            cache_key=f"standings_{league_id}_{season_id}.json",
        )
        
        team_mapping = {}
        for standings_table in data.get("data", []):
            for team_data in standings_table.get("standings", []):
                team_name = team_data.get("team_name")
                team_id = team_data.get("team_id")
                if team_name and team_id:
                    team_mapping[team_name] = team_id
                    
        return team_mapping
        
    except Exception as e:
        print(f"Error getting league teams: {e}")
        return {}


def fetch_teams(league_id: int, season_id: str) -> pd.DataFrame:
    """Fetch teams for a league/season."""
    team_mapping = get_league_teams(league_id, season_id)
    
    if not team_mapping:
        return pd.DataFrame(columns=["team", "team_id"])
    
    teams_data = [{"team": name, "team_id": tid} for name, tid in team_mapping.items()]
    return pd.DataFrame(teams_data)


def fetch_matches(league_id: int, season_id: str) -> pd.DataFrame:
    """Fetch match fixtures for a league/season."""
    
    try:
        data = _get(
            "/matches",
            {"league_id": league_id, "season_id": season_id},
            cache_key=f"matches_{league_id}_{season_id}.json",
        )
        
        matches = data.get("data", [])
        if not matches:
            return pd.DataFrame()
        
        df = pd.DataFrame(matches)
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
        
        df["season_id"] = season_id
        return df
        
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return pd.DataFrame()


def fetch_team_match_stats(team_id: str, league_id: int, season_id: str) -> pd.DataFrame:
    """Fetch detailed match statistics for a team."""
    
    try:
        data = _get(
            "/team-match-stats",
            {"team_id": team_id, "league_id": league_id, "season_id": season_id},
            cache_key=f"teamstats_{team_id}_{league_id}_{season_id}.json",
        )

        matches = data.get("data", [])
        if not matches:
            return pd.DataFrame()

        rows = []
        for match in matches:
            meta_data = match.get("meta_data", {})
            stats = match.get("stats", {})
            
            # Create combined row
            row = {**meta_data}
            
            # Flatten and add stats
            flattened_stats = _flatten_stats(stats)
            row.update(flattened_stats)
            
            rows.append(row)

        df = pd.DataFrame(rows)
        
        if df.empty:
            return df

        # Standardize important columns
        column_mapping = {
            "goals": "gf",
            "goals_against": "ga",
            "expected_goals": "xg", 
            "expected_goals_against": "xga",
            "shots_total": "shots",
            "shots_on_target": "sot",
            "possession": "poss",
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # Ensure core columns exist with defaults
        core_defaults = {
            "match_id": "",
            "date": pd.NaT,
            "team_id": team_id,
            "gf": 0,
            "ga": 0,
            "xg": 0.0,
            "xga": 0.0,
            "shots": 0,
            "sot": 0,
            "poss": 50.0,
            "home_away": "Home"
        }
        
        for col, default_val in core_defaults.items():
            if col not in df.columns:
                df[col] = default_val

        # Convert date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
            if df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)

        df["season_id"] = season_id
        return df
        
    except Exception as e:
        print(f"Error fetching team match stats for {team_id}: {e}")
        return pd.DataFrame()


def build_three_seasons(league_id: int, seasons: Iterable[str]) -> pd.DataFrame:
    """Build training dataset from multiple seasons."""
    
    print(f"Building dataset for league {league_id}, seasons: {list(seasons)}")
    
    all_seasons = []
    for season in seasons:
        try:
            season_data = _assemble_one_season(league_id, season)
            if not season_data.empty:
                all_seasons.append(season_data)
                print(f"Added {len(season_data)} matches from season {season}")
            else:
                print(f"No data for season {season}")
        except Exception as e:
            print(f"Failed to process season {season}: {e}")
            continue

    if not all_seasons:
        raise RuntimeError("No season data could be assembled")

    # Combine all seasons
    df = pd.concat(all_seasons, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"Combined dataset: {len(df)} matches")

    # Add rolling statistics
    df = _add_rolling_stats(df)
    
    # Add ELO ratings
    df = _add_elo_columns(df)

    # Add time-based weights
    df = _add_time_weights(df)

    return df


def _add_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling statistics for teams."""
    
    def add_roll(df_in: pd.DataFrame, side: str) -> pd.DataFrame:
        df_copy = df_in.sort_values("date").copy()
        team_col = f"team_id_{side}"
        
        if team_col not in df_copy.columns:
            print(f"Warning: {team_col} not found, skipping rolling stats")
            return df_copy
            
        grp = df_copy.groupby(team_col, dropna=False)
        
        for stat in ["xg", "xga", "shots", "sot", "gf", "ga", "poss"]:
            col = f"{stat}_{side}"
            if col not in df_copy.columns:
                df_copy[f"{stat}_roll{window}_{side}"] = np.nan
                continue
                
            try:
                rolled = grp[col].rolling(window, min_periods=max(3, window // 2)).mean()
                df_copy[f"{stat}_roll{window}_{side}"] = rolled.shift(1).reset_index(level=0, drop=True)
            except Exception as e:
                print(f"Error calculating rolling {stat} for {side}: {e}")
                df_copy[f"{stat}_roll{window}_{side}"] = np.nan
                
        return df_copy

    for side in ["H", "A"]:
        df = add_roll(df, side)
        
    return df


def _add_time_weights(df: pd.DataFrame, half_life_days: int = 240) -> pd.DataFrame:
    """Add time-based sample weights."""
    
    if "date" not in df.columns:
        df["sample_weight"] = 1.0
        return df
        
    try:
        dates = pd.to_datetime(df["date"])
        max_date = dates.max()
        
        if pd.isna(max_date):
            df["sample_weight"] = 1.0
            return df
            
        age_days = (max_date - dates).dt.days.clip(lower=0).astype(float)
        decay_rate = np.log(2) / max(half_life_days, 1)
        weights = np.exp(-decay_rate * age_days)
        
        # Normalize weights
        mean_weight = weights.mean()
        if mean_weight > 0:
            weights = weights / mean_weight
        else:
            weights = np.ones_like(weights)
            
        df["sample_weight"] = weights
        
    except Exception as e:
        print(f"Error calculating time weights: {e}")
        df["sample_weight"] = 1.0
        
    return df


def test_api_and_build_sample(league_id: int = 9, season_id: str = "2023-24") -> Optional[pd.DataFrame]:
    """Test the API and build a small sample dataset."""
    
    print("Testing API connection...")
    
    try:
        # Test API key
        api_key = _get_api_key()
        print(f"API key obtained: {'Yes' if api_key else 'No'}")
        
        # Test getting teams
        teams = fetch_teams(league_id, season_id)
        print(f"Found {len(teams)} teams")
        
        if teams.empty:
            print("No teams found - check league_id and season_id")
            return None
            
        # Test getting stats for one team
        if len(teams) > 0:
            first_team_id = teams.iloc[0]["team_id"]
            first_team_name = teams.iloc[0]["team"]
            print(f"Testing with team: {first_team_name} (ID: {first_team_id})")
            
            team_stats = fetch_team_match_stats(first_team_id, league_id, season_id)
            print(f"Found {len(team_stats)} matches for {first_team_name}")
            
            if not team_stats.empty:
                print("Sample columns:", list(team_stats.columns))
                return team_stats
                
        return None
        
    except Exception as e:
        print(f"API test failed: {e}")
        return None


__all__ = [
    "fetch_teams",
    "fetch_matches", 
    "fetch_team_match_stats",
    "build_three_seasons",
    "test_api_and_build_sample",
]
