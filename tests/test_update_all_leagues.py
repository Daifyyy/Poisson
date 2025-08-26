import sys
import pathlib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import utils.update_data as update_data


def test_update_all_leagues_adds_new_matches(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    existing_path = data_dir / "XX_combined_full_updated.csv"
    pd.DataFrame({
        "Date": ["01/08/2024"],
        "HomeTeam": ["A"],
        "AwayTeam": ["B"],
    }).to_csv(existing_path, index=False)

    monkeypatch.setattr(update_data, "LEAGUES", {"XX": "http://example.com/xx.csv"})

    new_csv = "Date,HomeTeam,AwayTeam\n01/08/2024,A,B\n08/08/2024,C,D\n"

    class DummyResp:
        def __init__(self, text):
            self.status_code = 200
            self.text = text

    monkeypatch.setattr(update_data.requests, "get", lambda url: DummyResp(new_csv))
    monkeypatch.chdir(tmp_path)

    messages = update_data.update_all_leagues()

    assert messages == ["✅ XX: Přidáno 1 nových zápasů."]

    updated = pd.read_csv(existing_path)
    assert len(updated) == 2
    assert "08/08/2024" in updated["Date"].values
