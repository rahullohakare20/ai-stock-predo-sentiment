import os
import torch
import numpy as np
from functools import lru_cache
from transformers import BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.config import settings

# FinBERT maps: 0=positive, 1=negative, 2=neutral
FINBERT_LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
FINBERT_SCORE_MAP = {0: 1.0, 1: -1.0, 2: 0.0}

MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")


@lru_cache(maxsize=1)
def load_finbert():
    """Lazy-load FinBERT — downloads once and caches locally."""
    print("[FinBERT] Loading model (first run downloads ~400MB)...")
    tokenizer = BertTokenizer.from_pretrained(
        "ProsusAI/finbert", cache_dir=MODEL_CACHE_DIR
    )
    model = BertForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", cache_dir=MODEL_CACHE_DIR
    )
    model.eval()
    print("[FinBERT] Model loaded.")
    return tokenizer, model


@lru_cache(maxsize=1)
def load_vader():
    return SentimentIntensityAnalyzer()


def score_vader(text: str) -> float:
    """Returns VADER compound score: -1 (most negative) to +1 (most positive)."""
    analyzer = load_vader()
    return analyzer.polarity_scores(text)["compound"]


def score_finbert(text: str) -> tuple[float, str]:
    """
    Returns (score, label) where score is mapped to -1 to +1.
    Truncates to 512 tokens (FinBERT limit).
    """
    tokenizer, model = load_finbert()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    predicted_class = torch.argmax(probs).item()
    label = FINBERT_LABEL_MAP[predicted_class]

    # Weighted score: positive_prob - negative_prob
    score = float(probs[0].item() - probs[1].item())  # pos - neg
    return round(score, 4), label


def score_ensemble(text: str) -> dict:
    """
    Runs both VADER and FinBERT and returns a combined ensemble result.
    Weights: FinBERT 70%, VADER 30% (FinBERT is finance-tuned).
    Falls back to VADER-only if FinBERT fails.
    """
    vader = score_vader(text)
    try:
        finbert_score, finbert_label = score_finbert(text)
        ensemble = round(0.7 * finbert_score + 0.3 * vader, 4)
    except Exception as e:
        print(f"[FinBERT] Inference failed, falling back to VADER: {e}")
        finbert_score = None
        finbert_label = None
        ensemble = vader

    signal = (
        "BULLISH" if ensemble >= 0.05
        else "BEARISH" if ensemble <= -0.05
        else "NEUTRAL"
    )

    return {
        "vader_score": round(vader, 4),
        "finbert_score": finbert_score,
        "finbert_label": finbert_label,
        "ensemble_score": ensemble,
        "signal": signal,
    }


def batch_score(items: list[dict]) -> list[dict]:
    """Score a list of scraper result dicts in-place. Returns enriched list."""
    for item in items:
        text = item.get("headline", "")
        if not text:
            continue
        scores = score_ensemble(text)
        item.update(scores)
    return items