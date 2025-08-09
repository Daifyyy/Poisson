import pytest
from utils.poisson_utils.match_style import intensity_score_to_emoji

def test_none_returns_blank():
    assert intensity_score_to_emoji(None) == ""

def test_invalid_type_returns_blank():
    assert intensity_score_to_emoji("oops") == ""

def test_value_ranges():
    assert intensity_score_to_emoji(1.2) == "🔥"
    assert intensity_score_to_emoji(0.5) == "⚡"
    assert intensity_score_to_emoji(0.0) == "➖"
    assert intensity_score_to_emoji(-0.5) == "❄️"
