"""Utility functions for navigation and query parameter handling."""

from typing import MutableMapping, Any


def clear_query_params_on_nav_change(
    session_state: MutableMapping[str, Any],
    query_params: MutableMapping[str, Any],
    navigation: str,
    view_param: str | None,
) -> None:
    """Clear stale query parameters when user switches app navigation.

    Parameters
    ----------
    session_state:
        Streamlit-like session state object used to track last navigation.
    query_params:
        Mutable mapping of the current query parameters (e.g. ``st.query_params``).
    navigation:
        The navigation label currently selected by the user.
    view_param:
        Value of the ``view`` query parameter if present.

    When navigation changes because the user clicked a link that already
    specifies ``view=match`` along with ``home_team`` and ``away_team``, these
    parameters must persist so the match prediction page can preselect the
    correct teams.  In all other cases the parameters are cleared to avoid
    stale selections.
    """

    link_triggered = view_param == "match" and navigation == "Match prediction"

    if "last_navigation" not in session_state:
        session_state["last_navigation"] = navigation
    elif session_state["last_navigation"] != navigation:
        session_state["last_navigation"] = navigation
        if not link_triggered:
            for param in ("selected_team", "home_team", "away_team", "view"):
                query_params.pop(param, None)
