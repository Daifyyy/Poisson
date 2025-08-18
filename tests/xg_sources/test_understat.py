from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.poisson_utils.xg_sources import understat as understat_module
from utils.poisson_utils.xg_sources.understat import fetch_understat_team_xg


FIXTURE = Path(__file__).with_name("understat_team_fixture.html")


def _load_fixture() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_fetch_understat_team_xg(monkeypatch, tmp_path):
    # isolate cache file
    cache_file = tmp_path / "understat_xg_cache.json"
    monkeypatch.setattr(understat_module, "CACHE_FILE", cache_file)

    html = _load_fixture()

    calls = {"count": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["count"] += 1
        class Resp:
            status_code = 200
            text = html
        return Resp()

    monkeypatch.setattr(understat_module.requests, "get", fake_get)

    data = fetch_understat_team_xg("TestTeam", "2020")
    assert data is not None
    assert data["xg"] == pytest.approx(1.1)
    assert data["xga"] == pytest.approx(0.8)
    assert cache_file.exists()
    assert calls["count"] == 1

    # subsequent call uses cache, no network
    def fail_get(*args, **kwargs):
        raise AssertionError("network call should not happen")

    monkeypatch.setattr(understat_module.requests, "get", fail_get)
    data_cached = fetch_understat_team_xg("TestTeam", "2020")
    assert data_cached == data
    assert calls["count"] == 1
