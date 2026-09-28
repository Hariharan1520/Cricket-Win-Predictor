"""
Database layer for persistent T20 match storage and historical replay (Phase 12).

Supports PostgreSQL in production via DATABASE_URL and automatically falls back
to SQLite (data/recent_matches.db) for local development and testing.
Uses SQLAlchemy 2.0 with psycopg v3.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# Load environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


class Base(DeclarativeBase):
    pass


class Match(Base):
    """Stores completed T20/T20I match metadata."""

    __tablename__ = "matches"

    match_id = Column(String(128), primary_key=True, index=True)
    name = Column(String(256), nullable=False)
    format = Column(String(32), nullable=False, default="T20")
    series_name = Column(String(256), nullable=True)
    venue = Column(String(256), nullable=True)
    city = Column(String(128), nullable=True)
    match_date = Column(String(64), nullable=True)
    date_time = Column(String(64), nullable=True)
    completed_at = Column(String(64), nullable=True)
    status = Column(String(256), nullable=True)
    winner = Column(String(128), nullable=True)
    result_text = Column(String(256), nullable=True)
    team_1 = Column(String(128), nullable=True)
    team_2 = Column(String(128), nullable=True)
    source = Column(String(64), nullable=False, default="cricket_api")
    source_updated_at = Column(DateTime, nullable=True)
    stored_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    analysis_available = Column(Boolean, default=False, nullable=False)
    innings_count = Column(Integer, default=0, nullable=False)

    deliveries = relationship("Delivery", back_populates="match", cascade="all, delete-orphan")
    match_states = relationship("MatchState", back_populates="match", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        teams_list = []
        if self.team_1:
            teams_list.append(self.team_1)
        if self.team_2:
            teams_list.append(self.team_2)

        return {
            "match_id": self.match_id,
            "name": self.name,
            "format": self.format,
            "match_type": self.format,
            "series_name": self.series_name,
            "venue": self.venue,
            "city": self.city,
            "date": self.match_date,
            "date_time": self.date_time or self.match_date,
            "status": self.status,
            "winner": self.winner,
            "result": self.result_text or self.status,
            "teams": teams_list,
            "team_1": self.team_1,
            "team_2": self.team_2,
            "completed_at": self.completed_at or self.date_time or self.match_date,
            "analysis_available": bool(self.analysis_available),
            "innings_count": self.innings_count,
            "created_at": self.stored_at.isoformat() if self.stored_at else None,
            "stored_at": self.stored_at.isoformat() if self.stored_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else (self.stored_at.isoformat() if self.stored_at else None),
        }


class Delivery(Base):
    """Stores ball-by-ball delivery event data."""

    __tablename__ = "deliveries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(128), ForeignKey("matches.match_id", ondelete="CASCADE"), nullable=False, index=True)
    innings = Column(Integer, nullable=False)
    over_number = Column(Integer, nullable=False)
    ball_number = Column(Integer, nullable=False)
    legal_ball_number = Column(Integer, nullable=False)
    batter = Column(String(128), nullable=True)
    bowler = Column(String(128), nullable=True)
    non_striker = Column(String(128), nullable=True)
    runs_batter = Column(Integer, default=0, nullable=False)
    runs_total = Column(Integer, default=0, nullable=False)
    extras = Column(Integer, default=0, nullable=False)
    wickets = Column(Integer, default=0, nullable=False)
    raw_info = Column(Text, nullable=True)

    match = relationship("Match", back_populates="deliveries")

    __table_args__ = (
        UniqueConstraint("match_id", "innings", "over_number", "ball_number", name="uq_match_delivery"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "match_id": self.match_id,
            "innings": self.innings,
            "over": self.over_number,
            "ball": self.ball_number,
            "legal_ball": self.legal_ball_number,
            "batter": self.batter,
            "bowler": self.bowler,
            "non_striker": self.non_striker,
            "runs_batter": self.runs_batter,
            "runs_total": self.runs_total,
            "extras": self.extras,
            "wickets": self.wickets,
        }


class MatchState(Base):
    """Stores exact 8 model feature vectors and evaluated probabilities per state."""

    __tablename__ = "match_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(128), ForeignKey("matches.match_id", ondelete="CASCADE"), nullable=False, index=True)
    innings = Column(Integer, nullable=False)
    legal_balls_completed = Column(Integer, nullable=False)
    overs_completed = Column(Float, nullable=False)
    target_score = Column(Float, nullable=False)
    current_score = Column(Float, nullable=False)
    wickets_lost = Column(Integer, nullable=False)
    runs_remaining = Column(Float, nullable=False)
    balls_remaining = Column(Integer, nullable=False)
    current_run_rate = Column(Float, nullable=False)
    required_run_rate = Column(Float, nullable=False)
    chasing_team_won = Column(Boolean, nullable=True)
    win_probability = Column(Float, nullable=False)
    probability_swing = Column(Float, default=0.0, nullable=False)

    match = relationship("Match", back_populates="match_states")

    __table_args__ = (
        UniqueConstraint("match_id", "innings", "legal_balls_completed", name="uq_match_state_ball"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "match_id": self.match_id,
            "innings": self.innings,
            "legal_balls_completed": self.legal_balls_completed,
            "overs_completed": round(self.overs_completed, 4),
            "target_score": self.target_score,
            "current_score": self.current_score,
            "wickets_lost": self.wickets_lost,
            "runs_remaining": self.runs_remaining,
            "balls_remaining": self.balls_remaining,
            "current_run_rate": round(self.current_run_rate, 2),
            "required_run_rate": round(self.required_run_rate, 2),
            "chasing_team_won": self.chasing_team_won,
            "win_probability": round(self.win_probability, 4),
            "win_probability_pct": round(self.win_probability * 100, 1),
            "probability_swing": round(self.probability_swing, 4),
        }


def get_database_url() -> str:
    """
    Retrieves and normalizes the database URL.
    Supports PostgreSQL via DATABASE_URL and defaults to local SQLite fallback.
    """
    raw_url = os.getenv("DATABASE_URL", "").strip()
    if not raw_url:
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        sqlite_path = data_dir / "recent_matches.db"
        return f"sqlite:///{sqlite_path}"

    # Normalize PostgreSQL URL for SQLAlchemy 2.0 with psycopg v3
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif raw_url.startswith("postgresql://") and not raw_url.startswith("postgresql+"):
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)

    return raw_url


def create_db_engine(database_url: Optional[str] = None):
    """Creates a SQLAlchemy engine configured for Neon PostgreSQL or SQLite fallback."""
    url = database_url or get_database_url()
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    # Optimized for Neon serverless PostgreSQL connection pooling & auto-reconnects
    return create_engine(url, pool_pre_ping=True, pool_recycle=300)


# Default engine & session factory
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(target_engine=None):
    """Initializes all database tables and ensures schema compatibility."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)

    # Safe migration for existing tables created with older schemas
    try:
        inspector = inspect(eng)
        if "matches" in inspector.get_table_names():
            columns = {c["name"] for c in inspector.get_columns("matches")}
            with eng.connect() as conn:
                if "date_time" not in columns:
                    conn.execute(text("ALTER TABLE matches ADD COLUMN date_time VARCHAR(64)"))
                if "completed_at" not in columns:
                    conn.execute(text("ALTER TABLE matches ADD COLUMN completed_at VARCHAR(64)"))
                if "updated_at" not in columns:
                    conn.execute(text("ALTER TABLE matches ADD COLUMN updated_at TIMESTAMP"))
                conn.commit()
    except Exception:
        pass


def get_db_session() -> Generator[Session, None, None]:
    """Yields a database session with guaranteed closure."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> Dict[str, Any]:
    """Checks database connectivity safely without exposing credentials."""
    url = get_database_url()
    db_type = "postgresql" if "postgresql" in url else "sqlite"
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"connected": True, "type": db_type}
    except Exception as e:
        return {"connected": False, "type": db_type, "error": str(e)}

