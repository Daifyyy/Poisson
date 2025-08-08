import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt


def poisson_prediction(home_exp, away_exp, max_goals=6):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i][j] = poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)
    return matrix


def match_outcomes_prob(matrix):
    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    return {
        'Home Win': round(home_win * 100, 2),
        'Draw': round(draw * 100, 2),
        'Away Win': round(away_win * 100, 2)
    }


def over_under_prob(matrix, line=2.5):
    prob_over = sum(
        matrix[i][j]
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
        if i + j > line
    )
    return {
        'Over 2.5': round(prob_over * 100, 2),
        'Under 2.5': round((1 - prob_over) * 100, 2)
    }


def btts_prob(matrix):
    btts = sum(
        matrix[i][j]
        for i in range(1, matrix.shape[0])
        for j in range(1, matrix.shape[1])
    )
    return {'BTTS Yes': round(btts * 100, 2), 'BTTS No': round((1 - btts) * 100, 2)}


def prob_to_odds(prob_percent):
    if prob_percent <= 0:
        return "∞"
    return round(100 / prob_percent, 2)


def generate_score_table_df(matrix, home_team, away_team):
    df = pd.DataFrame(matrix * 100)
    df.index.name = f"{home_team} góly"
    df.columns.name = f"{away_team} góly"
    styled = (
        df.style
        .format("{:.1f} %")
        .background_gradient(cmap="YlOrRd", axis=None)
        .set_properties(**{"text-align": "center", "font-size": "11px"})
    )
    return styled


def get_top_scorelines(matrix, top_n=5):
    score_probs = [((i, j), matrix[i][j]) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    score_probs.sort(key=lambda x: x[1], reverse=True)
    return score_probs[:top_n]


def plot_top_scorelines(score_probs, home_team, away_team):
    labels = [f"{a}:{b}" for (a, b), _ in score_probs]
    values = [round(p * 100, 2) for _, p in score_probs]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color='skyblue')
    ax.set_title(f"Top skóre: {home_team} vs {away_team}")
    ax.set_xlabel("Skóre")
    ax.set_ylabel("Pravděpodobnost (%)")
    return fig
