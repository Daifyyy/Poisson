from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.poisson_utils.xg_sources.pseudo import fetch_pseudo_xg


def test_fetch_pseudo_xg_structure():
    df = pd.DataFrame(
        {
            "HomeTeam": ["A", "B"],
            "AwayTeam": ["B", "A"],
            "HS": [10, 8],
            "HST": [5, 4],
            "AS": [8, 10],
            "AST": [4, 5],
            "FTHG": [1, 2],
            "FTAG": [0, 1],
        }
    )

    result = fetch_pseudo_xg("A", df)
    assert set(result.keys()) == {"xg", "xga"}
    assert isinstance(result["xg"], float)
    assert isinstance(result["xga"], float)
