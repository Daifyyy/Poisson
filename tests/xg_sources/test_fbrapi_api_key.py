import sys
from pathlib import Path

import types
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))

import utils.poisson_utils.xg_sources.fbrapi as fbrapi_module


def test_api_key_cached(monkeypatch, tmp_path):
    calls = {"post": 0}

    def fake_post(url, timeout=10):
        calls["post"] += 1
        class Resp:
            status_code = 200
            def json(self):
                return {"api_key": "TESTKEY"}
            def raise_for_status(self):
                pass
        return Resp()

    monkeypatch.setattr(fbrapi_module, "API_KEY_FILE", tmp_path / "key.txt")
    monkeypatch.setattr(fbrapi_module.requests, "post", fake_post)
    monkeypatch.delenv("FBRAPI_KEY", raising=False)

    key1 = fbrapi_module.get_fbrapi_api_key()
    key2 = fbrapi_module.get_fbrapi_api_key()

    assert key1 == "TESTKEY"
    assert key2 == "TESTKEY"
    assert calls["post"] == 1
