# src/app/db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import DATABASE_URL
from app.db.models import Base

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL, echo=False, future=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

def get_session() -> Session:
    """Helper untuk mendapatkan Session baru."""
    return SessionLocal()