import pandas as pd


def calculate_clean_sheets(df: pd.DataFrame, team: str) -> float:
    """Return the percentage of matches where the team kept a clean sheet.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing match results with columns 'HomeTeam', 'AwayTeam',
        'FTHG' (full-time home goals) and 'FTAG' (full-time away goals).
    team: str
        Team name to evaluate.

    Returns
    -------
    float
        Percentage of matches with a clean sheet rounded to one decimal place.
    """
    team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
    team_matches = team_matches.assign(
        is_clean_sheet=(
            ((team_matches["HomeTeam"] == team) & (team_matches["FTAG"] == 0))
            | ((team_matches["AwayTeam"] == team) & (team_matches["FTHG"] == 0))
        )
    )
    return (
        round(team_matches["is_clean_sheet"].mean() * 100, 1)
        if not team_matches.empty
        else 0
    )
