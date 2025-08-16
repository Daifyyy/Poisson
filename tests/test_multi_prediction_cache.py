import sections.multi_prediction_section as mp
from unittest.mock import patch
import pandas as pd
import utils.poisson_utils.xg as xg


def test_get_league_data_and_elo_uses_cache():
    league_file = "data/E0_combined_full_updated.csv"
    df = mp.load_data(league_file)
    match_list = [
        {
            "league_file": league_file,
            "league_name": "E0",
            "home_team": df.iloc[0]["HomeTeam"],
            "away_team": df.iloc[0]["AwayTeam"],
        },
        {
            "league_file": league_file,
            "league_name": "E0",
            "home_team": df.iloc[1]["HomeTeam"],
            "away_team": df.iloc[1]["AwayTeam"],
        },
    ]

    mp.get_league_data_and_elo.clear()

    with patch("sections.multi_prediction_section.load_data", wraps=mp.load_data) as mock_load, \
         patch("sections.multi_prediction_section.calculate_elo_ratings", wraps=mp.calculate_elo_ratings) as mock_elo:

        cached_results = []
        for match in match_list:
            df_match, elo_dict = mp.get_league_data_and_elo(match["league_file"])
            home_exp, away_exp = mp.expected_goals_weighted_by_elo(
                df_match, match["home_team"], match["away_team"], elo_dict
            )
            cached_results.append((home_exp, away_exp))

        assert mock_load.call_count == 1
        assert mock_elo.call_count == 1

    direct_results = []
    for match in match_list:
        df_match = mp.load_data(match["league_file"])
        elo_dict = mp.calculate_elo_ratings(df_match)
        home_exp, away_exp = mp.expected_goals_weighted_by_elo(
            df_match, match["home_team"], match["away_team"], elo_dict
        )
        direct_results.append((home_exp, away_exp))

    assert cached_results == direct_results


def test_expected_goals_weighted_by_elo_uses_cache():
    league_file = "data/E0_combined_full_updated.csv"
    df = mp.load_data(league_file)
    elo_dict = mp.calculate_elo_ratings(df)
    home_team = df.iloc[0]["HomeTeam"]
    away_team = df.iloc[0]["AwayTeam"]

    xg._expected_goals_cache.clear()
    xg._filtered_matches_cache.clear()
    xg._league_versions.clear()

    calls = {"count": 0}
    original_nsmallest = pd.DataFrame.nsmallest

    def wrapped_nsmallest(self, n, columns, keep="first"):
        calls["count"] += 1
        return original_nsmallest(self, n, columns, keep)

    with patch("pandas.DataFrame.nsmallest", new=wrapped_nsmallest):
        xg.expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
        first_call = calls["count"]
        xg.expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
        assert calls["count"] == first_call


def test_expected_goals_cache_invalidation_on_data_change():
    league_file = "data/E0_combined_full_updated.csv"
    df = mp.load_data(league_file)
    elo_dict = mp.calculate_elo_ratings(df)
    home_team = df.iloc[0]["HomeTeam"]
    away_team = df.iloc[0]["AwayTeam"]

    xg._expected_goals_cache.clear()
    xg._filtered_matches_cache.clear()
    xg._league_versions.clear()

    calls = {"count": 0}
    original_nsmallest = pd.DataFrame.nsmallest

    def wrapped_nsmallest(self, n, columns, keep="first"):
        calls["count"] += 1
        return original_nsmallest(self, n, columns, keep)

    with patch("pandas.DataFrame.nsmallest", new=wrapped_nsmallest):
        xg.expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
        first_count = calls["count"]
        xg.expected_goals_weighted_by_elo(df, home_team, away_team, elo_dict)
        assert calls["count"] == first_count

        # Modify league data -> cache should invalidate
        new_row = df.iloc[0].copy()
        new_row["Date"] = pd.to_datetime(df["Date"].max()) + pd.Timedelta(days=1)
        df_updated = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        elo_updated = mp.calculate_elo_ratings(df_updated)
        xg.expected_goals_weighted_by_elo(df_updated, home_team, away_team, elo_updated)
        assert calls["count"] > first_count
