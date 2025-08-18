from pathlib import Path
import sys
import types

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

import utils.poisson_utils.xg_sources as xg_sources


@pytest.fixture
def no_cache(monkeypatch):
    """Disable on-disk caching for provider chain."""
    monkeypatch.setattr(xg_sources, "_load_cache", lambda: {})
    monkeypatch.setattr(xg_sources, "_save_cache", lambda cache: None)


@pytest.fixture
def understat_success(no_cache, monkeypatch):
    us_mod = types.SimpleNamespace(
        get_team_xg_xga=lambda team, season: {"xg": 1.0, "xga": 2.0}
    )
    fb_mod = types.SimpleNamespace(
        get_team_xg_xga=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fbref should not be called")
        )
    )
    monkeypatch.setitem(sys.modules, "utils.poisson_utils.xg_sources.understat", us_mod)
    monkeypatch.setitem(sys.modules, "utils.poisson_utils.xg_sources.fbref", fb_mod)
    return xg_sources.get_team_xg_xga("Team", "2020")


@pytest.fixture
def fbref_fallback(no_cache, monkeypatch):
    us_mod = types.SimpleNamespace(
        get_team_xg_xga=lambda *args, **kwargs: (_ for _ in ()).throw(
            Exception("understat failure")
        )
    )
    fb_mod = types.SimpleNamespace(
        get_team_xg_xga=lambda team, season: {"xg": 3.0, "xga": 4.0}
    )
    monkeypatch.setitem(sys.modules, "utils.poisson_utils.xg_sources.understat", us_mod)
    monkeypatch.setitem(sys.modules, "utils.poisson_utils.xg_sources.fbref", fb_mod)
    return xg_sources.get_team_xg_xga("Team", "2020")


@pytest.fixture
def pseudo_fallback(no_cache, monkeypatch):
    fail_mod = types.SimpleNamespace(
        get_team_xg_xga=lambda *args, **kwargs: (_ for _ in ()).throw(
            Exception("provider failure")
        )
    )
    pseudo_mod = types.SimpleNamespace(
        fetch_pseudo_xg=lambda team, df: {"xg": 5.0, "xga": 6.0}
    )
    monkeypatch.setitem(sys.modules, "utils.poisson_utils.xg_sources.understat", fail_mod)
    monkeypatch.setitem(sys.modules, "utils.poisson_utils.xg_sources.fbref", fail_mod)
    monkeypatch.setitem(sys.modules, "utils.poisson_utils.xg_sources.pseudo", pseudo_mod)
    df = pd.DataFrame({"team": ["Team"], "x": [1]})
    return xg_sources.get_team_xg_xga("Team", "2020", league_df=df)


def test_understat_success(understat_success):
    result = understat_success
    assert result["source"] == "understat"
    assert result["xg"] == pytest.approx(1.0)
    assert result["xga"] == pytest.approx(2.0)


def test_fbref_fallback(fbref_fallback):
    result = fbref_fallback
    assert result["source"] == "fbref"
    assert result["xg"] == pytest.approx(3.0)
    assert result["xga"] == pytest.approx(4.0)


def test_pseudo_fallback(pseudo_fallback):
    result = pseudo_fallback
    assert result["source"] == "pseudo"
    assert result["xg"] == pytest.approx(5.0)
    assert result["xga"] == pytest.approx(6.0)
