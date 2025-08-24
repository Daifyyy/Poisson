from utils.navigation import clear_query_params_on_nav_change


def test_match_link_preserves_query_params():
    session_state = {"last_navigation": "League overview"}
    query_params = {"home_team": "Fulham", "away_team": "Man United", "view": "match"}
    clear_query_params_on_nav_change(session_state, query_params, "Match prediction", "match")
    assert query_params == {
        "home_team": "Fulham",
        "away_team": "Man United",
        "view": "match",
    }


def test_manual_navigation_clears_query_params():
    session_state = {"last_navigation": "Match prediction"}
    query_params = {"home_team": "Fulham", "away_team": "Man United", "view": "match"}
    clear_query_params_on_nav_change(session_state, query_params, "League overview", "match")
    assert query_params == {}
