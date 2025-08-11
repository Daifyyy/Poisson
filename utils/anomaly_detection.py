import numpy as np

def calculate_contrarian_risk_score(matrix, expected_goals, tempo_home, tempo_away, warning_index_home, warning_index_away):
    """
    Vyhodnotí riziko, že zápas nebude odpovídat očekávanému (např. místo nudného zápasu přestřelka).
    """

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

def calculate_confidence_index(
    outcomes: dict,
    poisson_matrix: np.ndarray,
    warning_home: float,
    warning_away: float,
    form_stability_score: float = 1.0,  # 1 = stabilní, <1 = nestabilní
    pos_home: float = 0.0,
    pos_away: float = 0.0,
    variance_warning: bool = False
) -> float:
    """
    Spočítá komplexní Confidence Index (0–100) na základě predikční dominance, variance, formy a rizik.
    """
    import numpy as np

    # Hybridní dominance score z 1X2 predikce
    sorted_probs = sorted(outcomes.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    ratio = sorted_probs[0] / sum(sorted_probs)
    hybrid_score = 1 * margin + 80 * ratio  # vážený dominance score

    # Rozptyl skóre z Poisson matice – čím větší variance, tím nižší jistota
    variance = np.var(poisson_matrix)
    variance_score = max(0, 30 - variance * 5)  # 0–25 bodů

    # Penalizace za variance warning (např. na základě šířky rozdělení)
    variance_penalty = 5 if variance_warning else 0
    
    # Penalizace za warning index (např. nízké ELO momentum, rizika)
    warning_penalty = 20 * max(warning_home, warning_away)

    # Bonus za pozitivní trendy (pokud např. tým má formu, momentum, atd.)
    trend_bonus = 40 * max(pos_home, pos_away)

    # Bonus nebo penalizace za stabilitu formy
    form_bonus = (form_stability_score - 1) * 50 # např. 0.95 → -1.0, 1.10 → +2.0

    # Výsledný confidence index
    confidence = (
        0.5 * hybrid_score +
        0.2 * variance_score +
        form_bonus +
        trend_bonus -
        warning_penalty -
        variance_penalty
    )
    
    # Horní posílení – když vše je ideální
    if confidence > 85 and not variance_warning and max(warning_home, warning_away) < 0.1:
        confidence = min(confidence + 10, 100)

    return round(np.clip(confidence, 0, 100), 1)
