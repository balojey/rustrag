import enum
import os
from pathlib import Path
from tempfile import gettempdir

from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = True
    llama_api_key: str = os.getenv("LLAMA_API_KEY")
    atlas_uri: str = os.getenv("ATLAS_URI")
    db_name: str = "rustrag"
    collection_name: str = "rust_docs"
    idx_name: str = "idx_embedding"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RUSTRAG_",
        env_file_encoding="utf-8",
    )


settings = Settings()
