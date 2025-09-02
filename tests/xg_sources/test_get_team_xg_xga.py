import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

import utils.poisson_utils.xg_sources as xg_sources
import utils.poisson_utils.xg_sources.understat as understat_module
import utils.poisson_utils.xg_sources.fbrapi as fbrapi_module
import utils.poisson_utils.xg_sources.fbref as fbref_module
from utils.poisson_utils.xg_sources import get_team_xg_xga


def test_get_team_xg_xga_understat(monkeypatch, tmp_path):
    monkeypatch.setattr(xg_sources, "CACHE_FILE", tmp_path / "xg_cache.json")
    calls = {"understat": 0, "fbrapi": 0, "fbref": 0}

    def fake_understat(team, season):
        calls["understat"] += 1
        return {"xg": 1.2, "xga": 0.9}

    def fake_fbrapi(team, season):
        calls["fbrapi"] += 1
        return {"xg": 0.7, "xga": 1.0}

    def fake_fbref(team, season):
        calls["fbref"] += 1
        return {"xg": 0.5, "xga": 0.6}

    monkeypatch.setattr(understat_module, "get_team_xg_xga", fake_understat)
    monkeypatch.setattr(fbrapi_module, "get_team_xg_xga", fake_fbrapi)
    monkeypatch.setattr(fbref_module, "get_team_xg_xga", fake_fbref)

    data = get_team_xg_xga("Test", "2020")
    assert data["source"] == "understat"
    assert data["xg"] == pytest.approx(1.2)
    assert data["xga"] == pytest.approx(0.9)
    assert calls["understat"] == 1
    assert calls["fbrapi"] == 0
    assert calls["fbref"] == 0


def test_get_team_xg_xga_fbrapi_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(xg_sources, "CACHE_FILE", tmp_path / "xg_cache.json")
    calls = {"understat": 0, "fbrapi": 0, "fbref": 0}

    def fail_understat(team, season):
        calls["understat"] += 1
        raise RuntimeError("fail")

    def fbrapi_success(team, season):
        calls["fbrapi"] += 1
        return {"xg": 0.7, "xga": 1.0}

    def fake_fbref(team, season):
        calls["fbref"] += 1
        return {"xg": 0.5, "xga": 0.6}

    monkeypatch.setattr(understat_module, "get_team_xg_xga", fail_understat)
    monkeypatch.setattr(fbrapi_module, "get_team_xg_xga", fbrapi_success)
    monkeypatch.setattr(fbref_module, "get_team_xg_xga", fake_fbref)

    data = get_team_xg_xga("Test", "2020")
    assert data["source"] == "fbrapi"
    assert data["xg"] == pytest.approx(0.7)
    assert data["xga"] == pytest.approx(1.0)
    assert calls["understat"] == 1
    assert calls["fbrapi"] == 1
    assert calls["fbref"] == 0


def test_get_team_xg_xga_fbref_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(xg_sources, "CACHE_FILE", tmp_path / "xg_cache.json")
    calls = {"understat": 0, "fbrapi": 0, "fbref": 0}

    def fail_understat(team, season):
        calls["understat"] += 1
        raise RuntimeError("fail")

    def fail_fbrapi(team, season):
        calls["fbrapi"] += 1
        raise RuntimeError("fail fbrapi")

    def fbref_success(team, season):
        calls["fbref"] += 1
        return {"xg": 0.8, "xga": 1.1}

    monkeypatch.setattr(understat_module, "get_team_xg_xga", fail_understat)
    monkeypatch.setattr(fbrapi_module, "get_team_xg_xga", fail_fbrapi)
    monkeypatch.setattr(fbref_module, "get_team_xg_xga", fbref_success)

    data = get_team_xg_xga("Test", "2020")
    assert data["source"] == "fbref"
    assert data["xg"] == pytest.approx(0.8)
    assert data["xga"] == pytest.approx(1.1)
    assert calls["understat"] == 1
    assert calls["fbrapi"] == 1
    assert calls["fbref"] == 1
