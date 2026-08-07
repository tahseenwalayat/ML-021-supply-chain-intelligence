import os
import yaml
from typing import Generator
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Helper to locate and read configs/config.yaml
def load_config(config_path: str = None) -> dict:
    """Locates and loads YAML configuration settings from configs/config.yaml."""
    if config_path is None:
        # Resolve config.yaml relative to workspace root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        config_path = os.path.join(base_dir, "configs", "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_database_url() -> str:
    """Build or retrieve database URL from environment variables or config.yaml."""
    # Priority 1: Direct DATABASE_URL environment variable
    env_db_url = os.getenv("DATABASE_URL")
    if env_db_url:
        return env_db_url

    # Priority 2: Build PostgreSQL URL from env vars or config.yaml
    cfg = load_config().get("database", {})
    host = os.getenv("POSTGRES_HOST", cfg.get("host", "localhost"))
    port = os.getenv("POSTGRES_PORT", str(cfg.get("port", 5432)))
    user = os.getenv("POSTGRES_USER", cfg.get("user", "postgres"))
    password = os.getenv("POSTGRES_PASSWORD", cfg.get("password", "postgres"))
    dbname = os.getenv("POSTGRES_DB", cfg.get("name", "supply_chain"))

    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_engine(url: str = None, pool_pre_ping: bool = True) -> Engine:
    """
    Returns an SQLAlchemy engine instance.

    Usage:
        from src.utils.db import get_engine
        engine = get_engine()
    """
    if url is None:
        url = get_database_url()

    # Configure connection pool settings for PostgreSQL vs SQLite
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(url, pool_pre_ping=pool_pre_ping, pool_size=10, max_overflow=20)

    return engine


def get_sessionmaker(engine: Engine = None) -> sessionmaker:
    """Returns an SQLAlchemy sessionmaker bound to the engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session() -> Generator[Session, None, None]:
    """Dependency helper yielding a database session context."""
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
