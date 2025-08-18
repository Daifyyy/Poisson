from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..xg import calculate_team_pseudo_xg


def fetch_pseudo_xg(team: str, df: pd.DataFrame) -> Dict[str, float]:
    """Compute pseudo xG and xGA for ``team`` using ``df`` league data."""
    stats = calculate_team_pseudo_xg(df)
    return stats.get(team, {})


def get_team_xg_xga(team: str, season: str, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Return pseudo-xG metrics for compatibility with xg_sources."""
    if df is None:
        return {}
    return fetch_pseudo_xg(team, df)
