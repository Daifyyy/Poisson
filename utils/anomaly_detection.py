def calculate_contrarian_risk_score(matrix, expected_goals, tempo_home, tempo_away, warning_index_home, warning_index_away):
    """
    Vyhodnotí riziko, že zápas nebude odpovídat očekávanému (např. místo nudného zápasu přestřelka).
    """
    import numpy as np

    risk_score = 0.0

    # Variance skóre (např. 2.5+ může značit přestřelku)
    variance = np.var(matrix)
    if variance > 2.0:
        risk_score += 0.3

    # Vysoké tempo obou týmů
    if tempo_home > 45 and tempo_away > 45:
        risk_score += 0.3

    # Nízký očekávaný počet gólů, ale vysoká variance => varování
    if expected_goals < 2.2 and variance > 1.8:
        risk_score += 0.2

    # Výstražné indexy
    if warning_index_home > 0.5 or warning_index_away > 0.5:
        risk_score += 0.2

    return min(risk_score, 1.0)


def calculate_upset_risk_score(outcomes, warning_index_fav, warning_index_dog, form_dog_positive):
    """
    Riziko, že outsider překvapí favorita.
    """
    risk_score = 0.0

    # Identifikace favorita a outsidera podle win probability
    if outcomes['Home Win'] > outcomes['Away Win']:
        fav = 'Home'
        dog = 'Away'
        fav_win_prob = outcomes['Home Win']
    else:
        fav = 'Away'
        dog = 'Home'
        fav_win_prob = outcomes['Away Win']

    if fav_win_prob > 60:
        risk_score += 0.2  # Vysoké očekávání
    
    if warning_index_fav > 0.5:
        risk_score += 0.3

    if warning_index_dog < 0.3 and form_dog_positive:
        risk_score += 0.3

    return min(risk_score, 1.0)


def colored_risk_tag(label, score):
    """Zobrazí barevný tag se skóre rizika."""
    if score < 0.3:
        color = "green"
    elif score < 0.6:
        color = "orange"
    else:
        color = "red"
    return f"<span style='background-color:{color}; color:white; padding:3px 8px; border-radius:10px;'>{label}: {int(score * 100)}%</span>"
