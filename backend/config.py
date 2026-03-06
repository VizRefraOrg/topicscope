import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str = os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://topicscope-openai.openai.azure.com/"
    )
    azure_openai_key: str = os.getenv("AZURE_OPENAI_KEY", "")
    azure_openai_embedding_deployment: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
    )
    azure_openai_api_version: str = "2024-06-01"

    # Azure AI Language
    azure_language_endpoint: str = os.getenv(
        "AZURE_LANGUAGE_ENDPOINT",
        "https://topicscope-lang.cognitiveservices.azure.com/"
    )
    azure_language_key: str = os.getenv("AZURE_LANGUAGE_KEY", "")

    # PostgreSQL
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://topicscopeadmin:Xx118899%40@topicscope.postgres.database.azure.com:5432/topicscope?sslmode=require"
    )

    # App settings
    cors_origins: list = ["http://localhost:3000", "https://topicscope-api.azurewebsites.net"]
    max_word_count: int = 10000

    class Config:
        env_file = ".env"


settings = Settings()
