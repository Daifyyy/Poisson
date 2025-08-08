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
    cs = 0
    for _, row in team_matches.iterrows():
        if row["HomeTeam"] == team and row["FTAG"] == 0:
            cs += 1
        elif row["AwayTeam"] == team and row["FTHG"] == 0:
            cs += 1
    return round(100 * cs / len(team_matches), 1) if len(team_matches) > 0 else 0
