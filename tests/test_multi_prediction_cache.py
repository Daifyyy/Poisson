import sections.multi_prediction_section as mp
from unittest.mock import patch

def test_preload_league_data_and_elo_reuses_results():
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

    with patch("sections.multi_prediction_section.load_data", wraps=mp.load_data) as mock_load, \
         patch("sections.multi_prediction_section.calculate_elo_ratings", wraps=mp.calculate_elo_ratings) as mock_elo:
        league_data_cache = {}
        elo_cache = {}
        for match in match_list:
            code = match["league_file"]
            if code not in league_data_cache:
                df_league = mp.load_data(code)
                mp.validate_dataset(df_league)
                league_data_cache[code] = df_league
                elo_cache[code] = mp.calculate_elo_ratings(df_league)
        assert mock_load.call_count == 1
        assert mock_elo.call_count == 1

        cached_results = []
        for match in match_list:
            df_match = league_data_cache[match["league_file"]]
            elo_dict = elo_cache[match["league_file"]]
            home_exp, away_exp = mp.expected_goals_weighted_by_elo(
                df_match, match["home_team"], match["away_team"], elo_dict
            )
            cached_results.append((home_exp, away_exp))

    direct_results = []
    for match in match_list:
        df_match = mp.load_data(match["league_file"])
        elo_dict = mp.calculate_elo_ratings(df_match)
        home_exp, away_exp = mp.expected_goals_weighted_by_elo(
            df_match, match["home_team"], match["away_team"], elo_dict
        )
        direct_results.append((home_exp, away_exp))

    assert cached_results == direct_results
