from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.poisson_utils.xg_sources import fbref as fbref_module
from utils.poisson_utils.xg_sources.fbref import fetch_fbref_team_xg

FIXTURE = Path(__file__).with_name("fbref_team_stats.html")


def _load_fixture() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_fetch_fbref_team_xg(monkeypatch, tmp_path):
    cache_file = tmp_path / "fbref_xg_cache.json"
    monkeypatch.setattr(fbref_module, "CACHE_FILE", cache_file)

    html = _load_fixture()
    calls = {"count": 0}

    class Resp:
        status_code = 200
        text = html

    def fake_get(url, headers=None, timeout=None):
        calls["count"] += 1
        return Resp()

    monkeypatch.setattr(fbref_module.requests, "get", fake_get)

    class FakeRobot:
        def set_url(self, url):
            pass
        def read(self):
            pass
        def can_fetch(self, ua, url):
            return True

    monkeypatch.setattr(fbref_module.robotparser, "RobotFileParser", FakeRobot)

    data = fetch_fbref_team_xg("Arsenal", "2023-2024", "9")
    assert data is not None
    assert data["xg"] == pytest.approx(70 / 38)
    assert data["xga"] == pytest.approx(40 / 38)
    assert cache_file.exists()
    assert calls["count"] == 1

    def fail_get(*args, **kwargs):
        raise AssertionError("network call should not happen")

    monkeypatch.setattr(fbref_module.requests, "get", fail_get)
    data_cached = fetch_fbref_team_xg("Arsenal", "2023-2024", "9")
    assert data_cached == data
    assert calls["count"] == 1
