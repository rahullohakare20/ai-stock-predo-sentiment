import feedparser
import yfinance as yf
import httpx
from datetime import datetime, timedelta
from typing import Optional
from app.config import settings

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False


RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.investing.com/rss/news_301.rss",
]


def _parse_rss_date(entry) -> datetime:
    """Try to parse a published date from feedparser entry."""
    try:
        t = entry.get("published_parsed") or entry.get("updated_parsed")
        if t:
            return datetime(*t[:6])
    except Exception:
        pass
    return datetime.utcnow()


class NewsScraper:
    """Fetches news headlines from multiple free sources."""

    def __init__(self):
        self.newsapi = None
        if NEWSAPI_AVAILABLE and settings.NEWS_API_KEY:
            self.newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY)

    def fetch_yfinance_news(self, ticker: str) -> list[dict]:
        """Uses yfinance built-in news — no API key needed."""
        results = []
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []
            for item in news[:20]:
                # yfinance news structure varies by version
                content = item.get("content", {})
                title = (
                    content.get("title")
                    or item.get("title")
                    or ""
                )
                url = (
                    content.get("canonicalUrl", {}).get("url")
                    or item.get("link")
                    or ""
                )
                pub_time = item.get("providerPublishTime") or item.get("published") or None
                published_at = (
                    datetime.utcfromtimestamp(pub_time)
                    if isinstance(pub_time, (int, float))
                    else datetime.utcnow()
                )
                if title:
                    results.append({
                        "ticker": ticker,
                        "source": "yfinance",
                        "headline": title,
                        "url": url,
                        "published_at": published_at,
                    })
        except Exception as e:
            print(f"[yfinance] Error fetching news for {ticker}: {e}")
        return results

    def fetch_newsapi(self, ticker: str, company_name: str = "") -> list[dict]:
        """Fetches from NewsAPI free tier (500 req/day)."""
        if not self.newsapi:
            return []
        results = []
        query = f"{ticker} stock" + (f" OR {company_name}" if company_name else "")
        try:
            response = self.newsapi.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                from_param=(datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
                page_size=20,
            )
            for article in response.get("articles", []):
                headline = article.get("title") or article.get("description") or ""
                if not headline or headline == "[Removed]":
                    continue
                pub = article.get("publishedAt", "")
                try:
                    published_at = datetime.fromisoformat(
                        pub.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    published_at = datetime.utcnow()
                results.append({
                    "ticker": ticker,
                    "source": "newsapi",
                    "headline": headline,
                    "url": article.get("url", ""),
                    "published_at": published_at,
                })
        except Exception as e:
            print(f"[newsapi] Error: {e}")
        return results

    def fetch_rss(self, ticker: str) -> list[dict]:
        """Parses RSS feeds filtering for ticker mentions."""
        results = []
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title} {summary}"
                    if ticker.upper() not in text.upper():
                        continue
                    results.append({
                        "ticker": ticker,
                        "source": "rss",
                        "headline": title,
                        "url": entry.get("link", ""),
                        "published_at": _parse_rss_date(entry),
                    })
            except Exception as e:
                print(f"[rss] Error parsing {url}: {e}")
        return results

    def fetch_all(self, ticker: str, company_name: str = "") -> list[dict]:
        """Aggregates all sources and deduplicates by headline."""
        all_items = (
            self.fetch_yfinance_news(ticker)
            + self.fetch_newsapi(ticker, company_name)
            + self.fetch_rss(ticker)
        )
        seen = set()
        unique = []
        for item in all_items:
            key = item["headline"][:80].lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        return unique


class RedditScraper:
    """
    Fetches Reddit posts using public JSON endpoints.
    No API key or credentials required.
    """

    SUBREDDITS = ["stocks", "investing", "wallstreetbets", "stockmarket", "finance"]

    def fetch(self, ticker: str, limit: int = 25) -> list[dict]:
        results = []
        headers = {"User-Agent": "StockPredoSentiment/1.0 (personal project)"}

        for sub in self.SUBREDDITS:
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={ticker}&sort=new&limit={limit}&t=week&restrict_sr=1"
            )
            try:
                r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
                if r.status_code != 200:
                    print(f"[reddit] r/{sub} returned {r.status_code}")
                    continue

                posts = r.json().get("data", {}).get("children", [])
                for post in posts:
                    d = post.get("data", {})
                    title = d.get("title", "").strip()
                    selftext = d.get("selftext", "").strip()
                    if not title:
                        continue
                    headline = title
                    if selftext and selftext not in ("[removed]", "[deleted]"):
                        headline += f" — {selftext[:200]}"
                    results.append({
                        "ticker": ticker,
                        "source": f"reddit/{sub}",
                        "headline": headline,
                        "url": f"https://reddit.com{d.get('permalink', '')}",
                        "published_at": datetime.utcfromtimestamp(
                            d.get("created_utc", datetime.utcnow().timestamp())
                        ),
                    })
            except Exception as e:
                print(f"[reddit] Error fetching r/{sub}: {e}")

        return results