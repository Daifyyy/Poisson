import sections.multi_prediction_section as mp
from unittest.mock import patch


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
