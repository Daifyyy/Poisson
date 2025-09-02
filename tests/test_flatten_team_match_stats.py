import fbrapi_dataset as fbd


def test_fetch_team_match_stats_flattens_nested_stats(monkeypatch):
    sample = {
        "data": [
            {
                "match_id": 1,
                "date": "2021-01-01",
                "team_id": 10,
                "opponent_id": 20,
                "team": "A",
                "gf": 1,
                "ga": 0,
                "xg": 0.5,
                "xga": 0.2,
                "shots": 5,
                "sot": 2,
                "poss": 55,
                "home_away": "home",
                "stats": {
                    "keeper": {"saves": 3},
                    "passing": {"pass": {"cmp": 100, "att": 110}},
                },
            }
        ]
    }

    def fake_get(path, params=None, cache_key=None):
        return sample

    monkeypatch.setattr(fbd, "_get", fake_get)
    df = fbd.fetch_team_match_stats(10, 1, "2021")

    assert "keeper_saves" in df.columns
    assert "passing_pass_cmp" in df.columns
    assert "passing_pass_att" in df.columns
    assert df.loc[0, "keeper_saves"] == 3
    assert df.loc[0, "passing_pass_cmp"] == 100
    assert df.loc[0, "passing_pass_att"] == 110
    assert df.loc[0, "season_id"] == "2021"
