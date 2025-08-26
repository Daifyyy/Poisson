"""Rule-based betting checklists.

This module implements three simple decision checklists used within the
application.  Each checklist evaluates a set of boolean rules based on a
supplied dictionary of statistics.  Every satisfied rule awards one point and
when the configured threshold is reached the checklist recommends the bet.

The functions are intentionally lightweight and do not depend on the rest of
repository code base.  They merely expect the caller to supply the required
statistics.  Missing values default to ``0`` which effectively means that the
rule is not satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

Rule = Tuple[str, Callable[[Dict[str, float]], bool]]


@dataclass
class ChecklistResult:
    """Result of a checklist evaluation.

    Attributes
    ----------
    score:
        Number of fulfilled rules.
    threshold:
        Required minimum score to recommend the bet.
    passed:
        ``True`` when ``score`` is greater or equal to ``threshold``.
    rule_results:
        Mapping of human readable rule descriptions to their boolean outcome.
    """

    score: int
    threshold: int
    passed: bool
    rule_results: Dict[str, bool]


def _evaluate_checklist(
    data: Dict[str, float], rules: Iterable[Rule], threshold: int
) -> ChecklistResult:
    """Evaluate ``rules`` on ``data`` and return a :class:`ChecklistResult`."""
    results: Dict[str, bool] = {}
    score = 0
    for description, func in rules:
        outcome = bool(func(data))
        results[description] = outcome
        if outcome:
            score += 1
    return ChecklistResult(score, threshold, score >= threshold, results)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_float(value: float | str | None) -> float:
    """Return ``value`` cast to ``float`` or ``0.0`` when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _select_stat(
    data: Dict[str, float], specific_key: str, season_key: str, matches_key: str
) -> float:
    """Return a statistic using a home/away value with seasonal fallback.

    When the number of matches for the specific venue is below five, seasonal
    averages are used as a fallback.  Missing keys default to ``0``.
    """
    if data.get(matches_key, 0) >= 5:
        return _to_float(data.get(specific_key))
    return _to_float(data.get(season_key))


# ---------------------------------------------------------------------------
# Checklist implementations
# ---------------------------------------------------------------------------

def over25_checklist(data: Dict[str, float], threshold: int = 7) -> ChecklistResult:
    """Checklist recommending an ``Over 2.5 goals`` bet."""
    rules: List[Rule] = [
        (
            "Home team scores >2.0 at home",
            lambda d: _select_stat(
                d, "home_goals_for_home_avg", "home_goals_for_season_avg", "home_matches_home"
            ) > 2.0,
        ),
        (
            "Away team scores >1.5 away",
            lambda d: _select_stat(
                d, "away_goals_for_away_avg", "away_goals_for_season_avg", "away_matches_away"
            ) > 1.5,
        ),
        (
            "Home defence concedes >1.0 at home",
            lambda d: _select_stat(
                d, "home_goals_against_home_avg", "home_goals_against_season_avg", "home_matches_home"
            ) > 1.0,
        ),
        (
            "Away defence concedes >1.0 away",
            lambda d: _select_stat(
                d, "away_goals_against_away_avg", "away_goals_against_season_avg", "away_matches_away"
            ) > 1.0,
        ),
        (
            "Both teams BTTS% >55%",
            lambda d: _to_float(d.get("btts_pct_home")) > 0.55
            and _to_float(d.get("btts_pct_away")) > 0.55,
        ),
        (
            "Both teams Over2.5% >55%",
            lambda d: _to_float(d.get("over25_pct_home")) > 0.55
            and _to_float(d.get("over25_pct_away")) > 0.55,
        ),
        (
            "Expected goals sum >2.8",
            lambda d: _to_float(d.get("poisson_home_exp"))
            + _to_float(d.get("poisson_away_exp"))
            > 2.8,
        ),
        (
            "Expected tempo >40",
            lambda d: _to_float(d.get("expected_tempo")) > 40,
        ),
        (
            "Both teams GII >0.3",
            lambda d: _to_float(d.get("gii_home")) > 0.3 and _to_float(d.get("gii_away")) > 0.3,
        ),
        (
            "Score variance >2.0",
            lambda d: _to_float(d.get("score_var")) > 2.0,
        ),
    ]
    return _evaluate_checklist(data, rules, threshold)


def home_win_checklist(data: Dict[str, float], threshold: int = 7) -> ChecklistResult:
    """Checklist recommending a ``Home win`` bet."""
    rules: List[Rule] = [
        ("Home >2 points per game (last5 at home)", lambda d: d.get("home_ppg_home5", 0) > 2.0),
        ("ELO home - away >= 50", lambda d: d.get("elo_home", 0) - d.get("elo_away", 0) >= 50),
        (
            "Home attack >1.8 at home",
            lambda d: _select_stat(
                d, "home_goals_for_home_avg", "home_goals_for_season_avg", "home_matches_home"
            ) > 1.8,
        ),
        (
            "Away defence concedes >1.5 away",
            lambda d: _select_stat(
                d, "away_goals_against_away_avg", "away_goals_against_season_avg", "away_matches_away"
            ) > 1.5,
        ),
        ("Home momentum positive", lambda d: d.get("momentum_home", 0) > 0),
        ("Away Warning Index >0.5", lambda d: d.get("warning_index_away", 0) > 0.5),
        ("Head-to-head home wins >50%", lambda d: d.get("h2h_home_win_pct", 0) > 0.5),
        ("Home advantage above league avg",
         lambda d: d.get("home_advantage_home", 0) > d.get("league_home_adv_avg", 0)),
        ("Last5 home ppg >=2", lambda d: d.get("home_ppg_home5", 0) >= 2.0),
        ("Last5 away ppg <1", lambda d: d.get("away_ppg_away5", 0) < 1.0),
    ]
    return _evaluate_checklist(data, rules, threshold)


def away_win_checklist(data: Dict[str, float], threshold: int = 7) -> ChecklistResult:
    """Checklist recommending an ``Away win`` bet."""
    rules: List[Rule] = [
        ("Away >2 points per game (last5 away)", lambda d: d.get("away_ppg_away5", 0) > 2.0),
        ("ELO away - home >= 50", lambda d: d.get("elo_away", 0) - d.get("elo_home", 0) >= 50),
        (
            "Away attack >1.8 away",
            lambda d: _select_stat(
                d, "away_goals_for_away_avg", "away_goals_for_season_avg", "away_matches_away"
            ) > 1.8,
        ),
        (
            "Home defence concedes >1.5 at home",
            lambda d: _select_stat(
                d, "home_goals_against_home_avg", "home_goals_against_season_avg", "home_matches_home"
            ) > 1.5,
        ),
        ("Away momentum positive", lambda d: d.get("momentum_away", 0) > 0),
        ("Home Warning Index >0.5", lambda d: d.get("warning_index_home", 0) > 0.5),
        ("Head-to-head away wins >50%", lambda d: d.get("h2h_away_win_pct", 0) > 0.5),
        ("Away advantage above league avg",
         lambda d: d.get("away_advantage_away", 0) > d.get("league_away_adv_avg", 0)),
        ("Last5 away ppg >=2", lambda d: d.get("away_ppg_away5", 0) >= 2.0),
        ("Last5 home ppg <1", lambda d: d.get("home_ppg_home5", 0) < 1.0),
    ]
    return _evaluate_checklist(data, rules, threshold)


__all__ = [
    "ChecklistResult",
    "over25_checklist",
    "home_win_checklist",
    "away_win_checklist",
]
