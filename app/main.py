from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
import yfinance as yf

from app.database import get_db, init_db
from app.models import SentimentRecord, TickerSummary
from app.scraper import NewsScraper, RedditScraper
from app.sentiment import batch_score

app = FastAPI(
    title="StockPredo Sentiment API",
    description="Real-time sentiment analysis for stocks using FinBERT + VADER",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

news_scraper = NewsScraper()
reddit_scraper = RedditScraper()


@app.on_event("startup")
def startup():
    init_db()
    print("Database initialized.")


# ─── Pydantic response schemas ──────────────────────────────────────────────

class SentimentResponse(BaseModel):
    ticker: str
    signal: str
    avg_score: float
    total_articles: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    updated_at: datetime

class ArticleResponse(BaseModel):
    id: int
    ticker: str
    source: str
    headline: str
    url: Optional[str]
    published_at: Optional[datetime]
    vader_score: Optional[float]
    finbert_score: Optional[float]
    finbert_label: Optional[str]
    ensemble_score: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Background task: scrape + score + persist ──────────────────────────────

def _run_analysis(ticker: str, db: Session):
    """Scrape all sources, score sentiment, save to DB, update summary."""
    ticker = ticker.upper()

    # 1. Scrape
    news_items = news_scraper.fetch_all(ticker)
    reddit_items = reddit_scraper.fetch(ticker)
    all_items = news_items + reddit_items

    if not all_items:
        return

    # 2. Score
    scored = batch_score(all_items)

    # 3. Persist articles (skip duplicates by headline)
    existing = {
        r.headline[:80] for r in
        db.query(SentimentRecord.headline)
        .filter(SentimentRecord.ticker == ticker)
        .all()
    }
    new_records = []
    for item in scored:
        key = item.get("headline", "")[:80]
        if key in existing:
            continue
        record = SentimentRecord(
            ticker=ticker,
            source=item.get("source", "unknown"),
            headline=item.get("headline", ""),
            url=item.get("url"),
            published_at=item.get("published_at"),
            vader_score=item.get("vader_score"),
            finbert_score=item.get("finbert_score"),
            finbert_label=item.get("finbert_label"),
            ensemble_score=item.get("ensemble_score"),
        )
        new_records.append(record)
        existing.add(key)

    db.add_all(new_records)
    db.commit()

    # 4. Compute summary stats
    all_ticker_records = (
        db.query(SentimentRecord)
        .filter(SentimentRecord.ticker == ticker)
        .filter(SentimentRecord.ensemble_score.isnot(None))
        .all()
    )
    scores = [r.ensemble_score for r in all_ticker_records]
    avg = sum(scores) / len(scores) if scores else 0.0
    bullish = sum(1 for s in scores if s >= 0.05)
    bearish = sum(1 for s in scores if s <= -0.05)
    neutral = len(scores) - bullish - bearish
    signal = "BULLISH" if avg >= 0.05 else "BEARISH" if avg <= -0.05 else "NEUTRAL"

    # Upsert summary
    summary = db.query(TickerSummary).filter(TickerSummary.ticker == ticker).first()
    if summary:
        summary.avg_score = round(avg, 4)
        summary.signal = signal
        summary.total_articles = len(scores)
        summary.bullish_count = bullish
        summary.bearish_count = bearish
        summary.neutral_count = neutral
        summary.updated_at = datetime.utcnow()
    else:
        summary = TickerSummary(
            ticker=ticker,
            avg_score=round(avg, 4),
            signal=signal,
            total_articles=len(scores),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
        )
        db.add(summary)
    db.commit()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "StockPredo-sentiment"}


@app.post("/analyze/{ticker}", response_model=SentimentResponse)
def analyze_ticker(
    ticker: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Triggers a fresh scrape + sentiment analysis for a ticker.
    Runs synchronously (for simplicity — can be backgrounded for prod).
    """
    ticker = ticker.upper()
    _run_analysis(ticker, db)

    summary = db.query(TickerSummary).filter(TickerSummary.ticker == ticker).first()
    if not summary:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    return SentimentResponse(
        ticker=summary.ticker,
        signal=summary.signal,
        avg_score=summary.avg_score,
        total_articles=summary.total_articles,
        bullish_count=summary.bullish_count,
        bearish_count=summary.bearish_count,
        neutral_count=summary.neutral_count,
        updated_at=summary.updated_at,
    )


@app.get("/sentiment/{ticker}", response_model=SentimentResponse)
def get_sentiment(ticker: str, db: Session = Depends(get_db)):
    """Returns cached sentiment summary without re-scraping."""
    ticker = ticker.upper()
    summary = db.query(TickerSummary).filter(TickerSummary.ticker == ticker).first()
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"No data for {ticker}. Call POST /analyze/{ticker} first.",
        )
    return SentimentResponse(
        ticker=summary.ticker,
        signal=summary.signal,
        avg_score=summary.avg_score,
        total_articles=summary.total_articles,
        bullish_count=summary.bullish_count,
        bearish_count=summary.bearish_count,
        neutral_count=summary.neutral_count,
        updated_at=summary.updated_at,
    )


@app.get("/articles/{ticker}", response_model=list[ArticleResponse])
def get_articles(
    ticker: str,
    limit: int = 50,
    source: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Returns individual articles with scores for a ticker."""
    ticker = ticker.upper()
    q = db.query(SentimentRecord).filter(SentimentRecord.ticker == ticker)
    if source:
        q = q.filter(SentimentRecord.source.ilike(f"%{source}%"))
    return q.order_by(desc(SentimentRecord.created_at)).limit(limit).all()


@app.get("/tickers")
def list_tickers(db: Session = Depends(get_db)):
    """Returns all tickers that have been analyzed."""
    summaries = db.query(TickerSummary).order_by(desc(TickerSummary.updated_at)).all()
    return [
        {
            "ticker": s.ticker,
            "signal": s.signal,
            "avg_score": s.avg_score,
            "total_articles": s.total_articles,
            "updated_at": s.updated_at,
        }
        for s in summaries
    ]


@app.get("/compare")
def compare_tickers(tickers: str, db: Session = Depends(get_db)):
    """Compare multiple tickers. Pass as ?tickers=AAPL,MSFT,TSLA"""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results = []
    for t in ticker_list:
        summary = db.query(TickerSummary).filter(TickerSummary.ticker == t).first()
        if summary:
            results.append({
                "ticker": summary.ticker,
                "signal": summary.signal,
                "avg_score": summary.avg_score,
                "total_articles": summary.total_articles,
            })
    return results