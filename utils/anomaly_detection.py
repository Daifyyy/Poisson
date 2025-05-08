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

def calculate_confidence_index(outcomes: dict, warning_home: float, warning_away: float,
                                pos_home: float, pos_away: float, variance_warning: bool) -> float:
    """
    Spočítá Confidence Index (0–100), který vyjadřuje, jak moc model věří své predikci.
    Vyšší hodnota = vyšší důvěra v jednoznačný výsledek a konzistenci vstupních metrik.
    """
    max_prob = max(outcomes.values())  # Např. 65 % Home Win
    sorted_probs = sorted(outcomes.values(), reverse=True)
    prob_diff = sorted_probs[0] - sorted_probs[1]  # Rozdíl mezi 1. a 2. nejpravděpodobnějším výsledkem

    # Penalizace za varování přestřelky
    variance_penalty = 0.1 if variance_warning else 0.0

    # Výpočet indexu
    confidence = (
        0.5 * (max_prob / 100) +                       # Dominance výsledku
        0.2 * (1 - max(warning_home, warning_away)) +  # Nízká rizika
        0.2 * max(pos_home, pos_away) -                # Pozitivní trendy
        variance_penalty                               # Penalizace za rozptyl výsledků
    )

    return round(min(max(confidence, 0), 1) * 100, 1)  # Výstup ve škále 0–100