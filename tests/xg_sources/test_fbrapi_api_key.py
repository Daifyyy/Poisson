import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import utils.poisson_utils.xg_sources.fbrapi as fbrapi_module


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("FBRAPI", "SECRET")
    key = fbrapi_module.get_fbrapi_api_key()
    assert key == "SECRET"


def test_api_key_missing(monkeypatch):
    monkeypatch.delenv("FBRAPI", raising=False)
    assert fbrapi_module.get_fbrapi_api_key() is None
