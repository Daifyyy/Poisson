import sys
from pathlib import Path
import math

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

import utils.poisson_utils.xg_sources as xg_sources
import utils.poisson_utils.xg_sources.fbrapi as fbrapi_module
from utils.poisson_utils.xg_sources import get_team_xg_xga


def test_get_team_xg_xga_fbrapi(monkeypatch, tmp_path):
    monkeypatch.setattr(xg_sources, "CACHE_FILE", tmp_path / "xg_cache.json")
    calls = {"fbrapi": 0}

    def fake_fbrapi(team, season):
        calls["fbrapi"] += 1
        return {"xg": 0.7, "xga": 1.0}

    monkeypatch.setattr(fbrapi_module, "get_team_xg_xga", fake_fbrapi)

    data = get_team_xg_xga("Test", "2020")
    assert data["source"] == "fbrapi"
    assert data["xg"] == pytest.approx(0.7)
    assert data["xga"] == pytest.approx(1.0)
    assert calls["fbrapi"] == 1

    # Second call uses cache
    data_cached = get_team_xg_xga("Test", "2020")
    assert data_cached == data
    assert calls["fbrapi"] == 1


def test_get_team_xg_xga_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(xg_sources, "CACHE_FILE", tmp_path / "xg_cache.json")

    def fail_fbrapi(team, season):
        raise RuntimeError("fail")

    monkeypatch.setattr(fbrapi_module, "get_team_xg_xga", fail_fbrapi)

    data = get_team_xg_xga("Test", "2020")
    assert data["source"] is None
    assert math.isnan(data["xg"])
    assert math.isnan(data["xga"])
