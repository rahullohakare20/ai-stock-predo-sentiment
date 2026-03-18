import pytest
from app.sentiment import score_vader, score_ensemble


def test_vader_positive():
    score = score_vader("Apple stock surges to record high on strong earnings")
    assert score > 0, "Expected positive VADER score for bullish headline"


def test_vader_negative():
    score = score_vader("Stock market crashes amid recession fears")
    assert score < 0, "Expected negative VADER score for bearish headline"


def test_vader_neutral():
    score = score_vader("Apple releases quarterly earnings report")
    assert -0.3 < score < 0.3, "Expected near-neutral VADER score"


def test_ensemble_returns_all_keys():
    result = score_ensemble("Tesla stock rises after earnings beat")
    assert "vader_score" in result
    assert "ensemble_score" in result
    assert "signal" in result
    assert result["signal"] in ("BULLISH", "BEARISH", "NEUTRAL")


def test_ensemble_score_range():
    for text in [
        "Massive stock crash wipes billions",
        "Neutral quarterly report released",
        "Record profits, strong buy signal",
    ]:
        result = score_ensemble(text)
        assert -1.0 <= result["ensemble_score"] <= 1.0