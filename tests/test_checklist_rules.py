from checklist_rules import over25_checklist


def test_over25_checklist_handles_invalid_gii() -> None:
    data = {"gii_home": "N/A", "gii_away": 0.4}
    result = over25_checklist(data)
    assert result.rule_results["Both teams GII >0.3"] is False


def test_over25_checklist_gii_rule_passes() -> None:
    data = {"gii_home": "0.4", "gii_away": 0.5}
    result = over25_checklist(data)
    assert result.rule_results["Both teams GII >0.3"] is True


def test_over25_checklist_handles_percentage_strings() -> None:
    data = {
        "btts_pct_home": "0.6",
        "btts_pct_away": 0.7,
        "over25_pct_home": "0.7",
        "over25_pct_away": "0.8",
    }
    result = over25_checklist(data)
    assert result.rule_results["Both teams BTTS% >55%"] is True
    assert result.rule_results["Both teams Over2.5% >55%"] is True
