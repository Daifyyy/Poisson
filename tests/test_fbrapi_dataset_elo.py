import pandas as pd
import pytest

from fbrapi_dataset import _add_elo_columns


def test_add_elo_columns_simple():
    df = pd.DataFrame(
        {
            "team_H": ["A", "B", "A"],
            "team_A": ["B", "A", "C"],
            "gf_H": [1, 0, 2],
            "gf_A": [0, 2, 1],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )

    res = _add_elo_columns(df)

    assert pytest.approx(res.loc[0, "elo_home"]) == 1500
    assert pytest.approx(res.loc[0, "elo_away"]) == 1500
    assert pytest.approx(res.loc[1, "elo_home"]) == 1490
    assert pytest.approx(res.loc[1, "elo_away"]) == 1510
    assert pytest.approx(res.loc[1, "elo_diff"]) == -20
    # After two matches, team A gained rating
    assert res.loc[2, "elo_home"] > res.loc[2, "elo_away"]
