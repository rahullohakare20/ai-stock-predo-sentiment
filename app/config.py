from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    NEWS_API_KEY: str = ""
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "StockPredoSentiment/1.0"
    DATABASE_URL: str = "sqlite:///./StockPredo.db"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    SENTIMENT_MODEL: str = "ensemble"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()