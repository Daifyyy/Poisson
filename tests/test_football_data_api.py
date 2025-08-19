import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.update_uefa_cups import fetch_matches, seasons_to_fetch


def _token() -> str:
    token = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token:
        pytest.skip("FOOTBALL_DATA_TOKEN not set")
    return token


def test_fetch_matches_returns_data():
    token = _token()
    season = max(seasons_to_fetch())
    try:
        df = fetch_matches("CL", season, token)
    except Exception as exc:  # pragma: no cover - network issues
        pytest.skip(f"API request failed: {exc}")
    assert not df.empty, "API returned no data"
