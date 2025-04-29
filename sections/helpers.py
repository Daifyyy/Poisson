# sections/helpers.py
def calculate_clean_sheets(team, df):
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    cs = 0
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team and row['FTAG'] == 0:
            cs += 1
        elif row['AwayTeam'] == team and row['FTHG'] == 0:
            cs += 1
    return round(100 * cs / len(team_matches), 1) if len(team_matches) > 0 else 0
