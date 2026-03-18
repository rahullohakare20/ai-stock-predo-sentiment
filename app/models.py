from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class SentimentRecord(Base):
    __tablename__ = "sentiment_records"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), index=True, nullable=False)
    source = Column(String(50), nullable=False)          # "news", "reddit", "rss"
    headline = Column(Text, nullable=False)
    url = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True)
    vader_score = Column(Float, nullable=True)           # compound -1 to 1
    finbert_score = Column(Float, nullable=True)         # mapped -1 to 1
    finbert_label = Column(String(20), nullable=True)    # positive/neutral/negative
    ensemble_score = Column(Float, nullable=True)        # final combined score
    created_at = Column(DateTime, default=datetime.utcnow)


class TickerSummary(Base):
    __tablename__ = "ticker_summaries"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), index=True, nullable=False)
    avg_score = Column(Float, nullable=False)
    signal = Column(String(10), nullable=False)          # BULLISH / BEARISH / NEUTRAL
    total_articles = Column(Integer, default=0)
    bullish_count = Column(Integer, default=0)
    bearish_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)