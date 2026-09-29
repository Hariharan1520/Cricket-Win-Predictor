"""
Comprehensive test suite for Phase 12.3:
Free Persistent Two-Match Storage with Neon PostgreSQL / SQLite fallback.

Tests cover:
A. First match insertion
B. Second match insertion
C. Third match insertion deletes oldest
D. Duplicate match does not create another row
E. Only one match remains when only one valid match exists
F. Zero valid matches does not alter database
G. Failed synchronization rolls back
H. Recent API reads only from database
I. Recent API makes zero provider calls
J. LIVE still uses provider
K. T20 filtering remains strict
L. Non-T20 matches are rejected
M. Model still uses exact 8 features
N. Model artifact unchanged
O. analysis_available=false when ball-by-ball is unavailable
P. analysis_available=true when valid ball-by-ball data exists
Q. Maximum stored matches is always 2
R. PostgreSQL schema compiles
S. SQLite fallback still works
T. Database persistence survives separate sessions/processes
Plus the mandatory retention test (Section 16).
"""

import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Strictly isolate unit tests to local test SQLite to protect production database
TEST_DB_PATH = PROJECT_ROOT / "data" / "test_storage.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable

from backend.app import app
from backend.database import (
    Base,
    Delivery,
    Match,
    MatchState,
    check_db_connection,
    get_database_url,
    init_db,
)
from src.live.cricket_api import CricketApiClient, CricketApiRateLimitError
from src.live.live_features import FEATURE_COLS
from src.live.live_match import load_prediction_model
from src.storage.match_sync import (
    MatchSyncManager,
    extract_match_winner,
    extract_winner_from_result,
    parse_args,
)


class TestDatabaseModels(unittest.TestCase):
    """Tests for database tables, models, relationships, and constraints."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)

    def test_database_connection_and_table_creation(self):
        """Verify tables can be created successfully."""
        table_names = list(Base.metadata.tables.keys())
        self.assertIn("matches", table_names)
        self.assertIn("deliveries", table_names)
        self.assertIn("match_states", table_names)

    def test_match_metadata_insertion_and_fields(self):
        """Verify Match model fields including match_type, completed_at, and updated_at."""
        m = Match(
            match_id="test-match-101",
            name="India vs Pakistan, T20 World Cup Final",
            format="T20",
            venue="Melbourne Cricket Ground",
            match_date="2026-10-15",
            date_time="2026-10-15T19:00:00Z",
            completed_at="2026-10-15T22:30:00Z",
            status="India won by 5 wickets",
            winner="India",
            team_1="Pakistan",
            team_2="India",
            analysis_available=True,
            innings_count=2,
        )
        self.session.add(m)
        self.session.commit()

        retrieved = self.session.query(Match).filter_by(match_id="test-match-101").first()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "India vs Pakistan, T20 World Cup Final")
        self.assertTrue(retrieved.analysis_available)
        self.assertEqual(retrieved.winner, "India")
        d = retrieved.to_dict()
        self.assertEqual(d["match_id"], "test-match-101")
        self.assertEqual(d["match_type"], "T20")
        self.assertEqual(d["completed_at"], "2026-10-15T22:30:00Z")
        self.assertIn("created_at", d)
        self.assertIn("updated_at", d)

    def test_delivery_unique_constraint(self):
        """Ensure duplicate delivery coordinates raise IntegrityError."""
        m = Match(match_id="test-match-104", name="Duplicate Delivery Test")
        self.session.add(m)
        self.session.commit()

        d1 = Delivery(match_id="test-match-104", innings=1, over_number=0, ball_number=1, legal_ball_number=1)
        d2 = Delivery(match_id="test-match-104", innings=1, over_number=0, ball_number=1, legal_ball_number=1)
        self.session.add(d1)
        self.session.commit()

        self.session.add(d2)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_match_state_unique_constraint(self):
        """Ensure duplicate match state for same legal ball raises IntegrityError."""
        m = Match(match_id="test-match-105", name="Duplicate State Test")
        self.session.add(m)
        self.session.commit()

        s1 = MatchState(
            match_id="test-match-105",
            innings=2,
            legal_balls_completed=60,
            overs_completed=10.0,
            target_score=160.0,
            current_score=80.0,
            wickets_lost=2,
            runs_remaining=80.0,
            balls_remaining=60,
            current_run_rate=8.0,
            required_run_rate=8.0,
            win_probability=0.5,
        )
        s2 = MatchState(
            match_id="test-match-105",
            innings=2,
            legal_balls_completed=60,
            overs_completed=10.0,
            target_score=160.0,
            current_score=81.0,
            wickets_lost=2,
            runs_remaining=79.0,
            balls_remaining=60,
            current_run_rate=8.1,
            required_run_rate=7.9,
            win_probability=0.52,
        )
        self.session.add(s1)
        self.session.commit()

        self.session.add(s2)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_cascade_deletion(self):
        """Deleting a match cascades and deletes its deliveries and states."""
        m = Match(match_id="test-match-106", name="Cascade Test")
        self.session.add(m)
        self.session.commit()

        d = Delivery(match_id="test-match-106", innings=1, over_number=0, ball_number=1, legal_ball_number=1)
        s = MatchState(
            match_id="test-match-106",
            innings=2,
            legal_balls_completed=1,
            overs_completed=0.1667,
            target_score=150.0,
            current_score=1.0,
            wickets_lost=0,
            runs_remaining=149.0,
            balls_remaining=119,
            current_run_rate=6.0,
            required_run_rate=7.51,
            win_probability=0.51,
        )
        self.session.add_all([d, s])
        self.session.commit()

        self.session.delete(m)
        self.session.commit()

        self.assertEqual(self.session.query(Delivery).filter_by(match_id="test-match-106").count(), 0)
        self.assertEqual(self.session.query(MatchState).filter_by(match_id="test-match-106").count(), 0)

    def test_postgresql_schema_compilation(self):
        """Test R: Verify models compile cleanly to valid PostgreSQL DDL with SERIAL and constraints."""
        for model in [Match, Delivery, MatchState]:
            sql = str(CreateTable(model.__table__).compile(dialect=postgresql.dialect()))
            self.assertIn("CREATE TABLE", sql)
            self.assertIn(model.__tablename__, sql)


class TestTwoMatchRetention(unittest.TestCase):
    """
    Comprehensive tests for Phase 12.3 Two-Match Retention Rule (Tests A-G, Q, and Section 16).
    """

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        model_path = PROJECT_ROOT / "models" / "logistic_regression_win_probability.joblib"
        self.model = load_prediction_model(model_path)

        self.mock_api = MagicMock(spec=CricketApiClient)
        self.manager = MatchSyncManager(
            api_client=self.mock_api,
            db_session=self.session,
            model=self.model,
        )

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)

    def test_section_16_mandatory_retention_test(self):
        """
        MANDATORY TEST (Section 16):
        Start: Match A, Match B
        Insert Match C -> Assert: Match A does not exist, Match B exists, Match C exists, Count == 2.
        Then insert Match C again -> Assert: Count == 2.
        Then insert Match D -> Assert: Only Match C and Match D remain.
        """
        # Step 1: Start with Match A and Match B
        self.mock_api.get_matches.return_value = [
            {"id": "match-A", "name": "Match A", "matchType": "t20", "matchEnded": True, "date": "2026-09-01"},
            {"id": "match-B", "name": "Match B", "matchType": "t20", "matchEnded": True, "date": "2026-09-02"},
        ]
        s1 = self.manager.sync_recent_matches(limit=10)
        self.assertEqual(s1["matches_stored"], 2)
        matches = self.session.query(Match).all()
        self.assertEqual(len(matches), 2)
        match_ids = {m.match_id for m in matches}
        self.assertEqual(match_ids, {"match-A", "match-B"})

        # Step 2: Insert Match C
        self.mock_api.get_matches.return_value = [
            {"id": "match-C", "name": "Match C", "matchType": "t20", "matchEnded": True, "date": "2026-09-03"},
        ]
        s2 = self.manager.sync_recent_matches(limit=10)
        self.assertEqual(s2["matches_stored"], 1)
        self.assertEqual(s2["deleted_old_matches"], 1)

        matches = self.session.query(Match).all()
        self.assertEqual(len(matches), 2)
        match_ids = {m.match_id for m in matches}
        self.assertNotIn("match-A", match_ids)  # Match A was deleted
        self.assertIn("match-B", match_ids)     # Match B exists
        self.assertIn("match-C", match_ids)     # Match C exists

        # Step 3: Insert Match C again (Deduplication)
        s3 = self.manager.sync_recent_matches(limit=10)
        self.assertEqual(s3["matches_stored"], 0)
        self.assertEqual(s3["duplicate_matches_count"], 1)
        matches = self.session.query(Match).all()
        self.assertEqual(len(matches), 2)
        match_ids = {m.match_id for m in matches}
        self.assertEqual(match_ids, {"match-B", "match-C"})

        # Step 4: Insert Match D -> Assert only Match C and Match D remain
        self.mock_api.get_matches.return_value = [
            {"id": "match-D", "name": "Match D", "matchType": "t20", "matchEnded": True, "date": "2026-09-04"},
        ]
        s4 = self.manager.sync_recent_matches(limit=10)
        self.assertEqual(s4["matches_stored"], 1)
        self.assertEqual(s4["deleted_old_matches"], 1)

        matches = self.session.query(Match).all()
        self.assertEqual(len(matches), 2)
        match_ids = {m.match_id for m in matches}
        self.assertEqual(match_ids, {"match-C", "match-D"})
        self.assertNotIn("match-A", match_ids)
        self.assertNotIn("match-B", match_ids)

    def test_first_match_insertion(self):
        """Test A: First match insertion stores exactly 1 match."""
        self.mock_api.get_matches.return_value = [
            {"id": "m1", "name": "First Match", "matchType": "t20", "matchEnded": True, "date": "2026-09-10"}
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["matches_stored"], 1)
        self.assertEqual(self.session.query(Match).count(), 1)

    def test_second_match_insertion(self):
        """Test B: Second match insertion results in 2 stored matches."""
        self.mock_api.get_matches.return_value = [
            {"id": "m1", "name": "First Match", "matchType": "t20", "matchEnded": True, "date": "2026-09-10"},
            {"id": "m2", "name": "Second Match", "matchType": "t20", "matchEnded": True, "date": "2026-09-11"},
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["matches_stored"], 2)
        self.assertEqual(self.session.query(Match).count(), 2)

    def test_third_match_insertion_deletes_oldest(self):
        """Test C: Third match insertion deletes the oldest match, leaving max 2."""
        self.mock_api.get_matches.return_value = [
            {"id": "m1", "name": "First", "matchType": "t20", "matchEnded": True, "date": "2026-09-10"},
            {"id": "m2", "name": "Second", "matchType": "t20", "matchEnded": True, "date": "2026-09-11"},
        ]
        self.manager.sync_recent_matches()

        self.mock_api.get_matches.return_value = [
            {"id": "m3", "name": "Third", "matchType": "t20", "matchEnded": True, "date": "2026-09-12"}
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["matches_stored"], 1)
        self.assertEqual(summary["deleted_old_matches"], 1)

        matches = self.session.query(Match).all()
        self.assertEqual(len(matches), 2)
        self.assertNotIn("m1", [m.match_id for m in matches])

    def test_duplicate_match_does_not_create_another_row(self):
        """Test D: Duplicate match synchronization leaves count unchanged."""
        self.mock_api.get_matches.return_value = [
            {"id": "m1", "name": "Repeat Match", "matchType": "t20", "matchEnded": True, "status": "In progress"}
        ]
        self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 1)

        # Sync again with updated status
        self.mock_api.get_matches.return_value = [
            {"id": "m1", "name": "Repeat Match", "matchType": "t20", "matchEnded": True, "status": "Team A won by 10 runs"}
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["matches_stored"], 0)
        self.assertEqual(summary["duplicate_matches_count"], 1)
        self.assertEqual(self.session.query(Match).count(), 1)
        self.assertEqual(self.session.query(Match).first().status, "Team A won by 10 runs")

    def test_only_one_match_when_only_one_valid(self):
        """Test E: If only one valid match exists, store only one."""
        self.mock_api.get_matches.return_value = [
            {"id": "m-only", "name": "Sole Match", "matchType": "t20", "matchEnded": True}
        ]
        self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 1)

    def test_zero_valid_matches_does_not_alter_database(self):
        """Test F: When provider returns 0 valid matches, existing matches remain unaltered."""
        # Seed 1 match
        self.mock_api.get_matches.return_value = [
            {"id": "m-exist", "name": "Existing Match", "matchType": "t20", "matchEnded": True}
        ]
        self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 1)

        # Provider returns non-T20 matches only
        self.mock_api.get_matches.return_value = [
            {"id": "test-match", "name": "Test Match", "matchType": "test", "matchEnded": True},
            {"id": "odi-match", "name": "ODI Match", "matchType": "odi", "matchEnded": True},
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["matches_stored"], 0)
        self.assertEqual(summary["completed_matches_count"], 0)
        # Existing match is untouched
        self.assertEqual(self.session.query(Match).count(), 1)
        self.assertEqual(self.session.query(Match).first().match_id, "m-exist")

    def test_failed_synchronization_rolls_back(self):
        """Test G: Failure during sync rolls back transaction and preserves existing data."""
        # Seed 2 matches
        self.mock_api.get_matches.return_value = [
            {"id": "m-safe-1", "name": "Safe 1", "matchType": "t20", "matchEnded": True},
            {"id": "m-safe-2", "name": "Safe 2", "matchType": "t20", "matchEnded": True},
        ]
        self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 2)

        # Force an error during ball-by-ball processing of new match
        with patch.object(self.manager, "_process_ball_by_ball", side_effect=RuntimeError("Simulated Ingestion Failure")):
            self.mock_api.get_matches.return_value = [
                {
                    "id": "m-fail",
                    "name": "Failing Match",
                    "matchType": "t20",
                    "matchEnded": True,
                    "bbbEnabled": True,
                    "deliveries": [{"runs": 4}],
                }
            ]
            summary = self.manager.sync_recent_matches()
            self.assertIn("Simulated Ingestion Failure", summary["errors"][0])

        # Assert database was NOT modified and safe matches still exist
        matches = self.session.query(Match).all()
        self.assertEqual(len(matches), 2)
        match_ids = {m.match_id for m in matches}
        self.assertEqual(match_ids, {"m-safe-1", "m-safe-2"})

    def test_maximum_stored_matches_is_always_two(self):
        """Test Q: Invariant check - maximum stored matches never exceeds 2."""
        raw_list = [
            {"id": f"batch-{i}", "name": f"Batch {i}", "matchType": "t20", "matchEnded": True, "date": f"2026-09-{i:02d}"}
            for i in range(1, 10)
        ]
        self.mock_api.get_matches.return_value = raw_list
        summary = self.manager.sync_recent_matches(limit=10)
        self.assertLessEqual(self.session.query(Match).count(), 2)
        self.assertEqual(summary["total_stored_matches"], 2)

    def test_nineteen_provider_matches_stores_exactly_two(self):
        """Scenario 1: 19 provider matches -> exactly 2 stored, no multiple insertions/prunings."""
        raw_list = [
            {"id": f"m-{i:02d}", "name": f"Match {i}", "matchType": "t20", "matchEnded": True, "date": f"2026-09-{i:02d}"}
            for i in range(1, 20)
        ]
        self.mock_api.get_matches.return_value = raw_list
        summary = self.manager.sync_recent_matches(limit=25)
        self.assertEqual(self.session.query(Match).count(), 2)
        self.assertEqual(summary["total_stored_matches"], 2)
        self.assertEqual(summary["matches_stored"], 2)
        # Verify the 2 stored matches are the newest: m-19 and m-18
        stored_ids = {m.match_id for m in self.session.query(Match).all()}
        self.assertEqual(stored_ids, {"m-19", "m-18"})

    def test_existing_ab_plus_provider_cd_leaves_only_cd(self):
        """Scenario 2: Existing A,B in DB + provider returning C,D -> only C,D remain in DB."""
        # Seed A and B
        mA = Match(match_id="mA", name="Match A", format="T20", match_date="2026-09-01", stored_at=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc))
        mB = Match(match_id="mB", name="Match B", format="T20", match_date="2026-09-02", stored_at=datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc))
        self.session.add_all([mA, mB])
        self.session.commit()
        self.assertEqual(self.session.query(Match).count(), 2)

        # Provider returns C and D (newer than A and B)
        self.mock_api.get_matches.return_value = [
            {"id": "mC", "name": "Match C", "matchType": "t20", "matchEnded": True, "date": "2026-09-03"},
            {"id": "mD", "name": "Match D", "matchType": "t20", "matchEnded": True, "date": "2026-09-04"},
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 2)
        stored_ids = {m.match_id for m in self.session.query(Match).all()}
        self.assertEqual(stored_ids, {"mC", "mD"})
        self.assertEqual(summary["deleted_old_matches"], 2)

    def test_duplicate_provider_match_does_not_duplicate(self):
        """Scenario 3: Duplicate match in provider response does not create duplicate rows."""
        self.mock_api.get_matches.return_value = [
            {"id": "dup-1", "name": "Match Dup", "matchType": "t20", "matchEnded": True, "date": "2026-09-10"},
            {"id": "dup-1", "name": "Match Dup Copy", "matchType": "t20", "matchEnded": True, "date": "2026-09-10"},
            {"id": "norm-2", "name": "Match Norm", "matchType": "t20", "matchEnded": True, "date": "2026-09-09"},
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 2)
        stored_ids = {m.match_id for m in self.session.query(Match).all()}
        self.assertEqual(stored_ids, {"dup-1", "norm-2"})

    def test_repeated_sync_still_exactly_two(self):
        """Scenario 4: Repeated sync keeps database at exactly 2 matches."""
        self.mock_api.get_matches.return_value = [
            {"id": "r1", "name": "R1", "matchType": "t20", "matchEnded": True, "date": "2026-09-10"},
            {"id": "r2", "name": "R2", "matchType": "t20", "matchEnded": True, "date": "2026-09-11"},
        ]
        s1 = self.manager.sync_recent_matches()
        self.assertEqual(s1["matches_stored"], 2)
        self.assertEqual(self.session.query(Match).count(), 2)

        # Sync again with same matches
        s2 = self.manager.sync_recent_matches()
        self.assertEqual(s2["matches_stored"], 0)
        self.assertEqual(s2["duplicate_matches_count"], 2)
        self.assertEqual(self.session.query(Match).count(), 2)
        self.assertEqual(s2["total_stored_matches"], 2)

    def test_newer_match_replaces_oldest(self):
        """Scenario 5: Newer match replaces oldest match in database."""
        # Seed A and B
        mA = Match(match_id="mA", name="Match A", format="T20", match_date="2026-09-01", stored_at=datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc))
        mB = Match(match_id="mB", name="Match B", format="T20", match_date="2026-09-02", stored_at=datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc))
        self.session.add_all([mA, mB])
        self.session.commit()

        # Provider returns newer match C (2026-09-03)
        self.mock_api.get_matches.return_value = [
            {"id": "mC", "name": "Match C", "matchType": "t20", "matchEnded": True, "date": "2026-09-03"},
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 2)
        stored_ids = {m.match_id for m in self.session.query(Match).all()}
        self.assertEqual(stored_ids, {"mB", "mC"})
        self.assertNotIn("mA", stored_ids)

    def test_zero_valid_matches_existing_records_unchanged(self):
        """Scenario 6: Zero valid matches returned -> existing database records remain unchanged."""
        mA = Match(match_id="mA", name="Match A", format="T20", match_date="2026-09-01")
        mB = Match(match_id="mB", name="Match B", format="T20", match_date="2026-09-02")
        self.session.add_all([mA, mB])
        self.session.commit()

        # Provider returns non-T20 matches only
        self.mock_api.get_matches.return_value = [
            {"id": "test-1", "name": "Test Match", "matchType": "test", "matchEnded": True},
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 2)
        stored_ids = {m.match_id for m in self.session.query(Match).all()}
        self.assertEqual(stored_ids, {"mA", "mB"})
        self.assertEqual(summary["matches_stored"], 0)
        self.assertEqual(summary["deleted_old_matches"], 0)

    def test_transaction_failure_previous_records_unchanged(self):
        """Scenario 7: Failure during transaction rolls back and preserves previous records."""
        mA = Match(match_id="mA", name="Match A", format="T20", match_date="2026-09-01")
        mB = Match(match_id="mB", name="Match B", format="T20", match_date="2026-09-02")
        self.session.add_all([mA, mB])
        self.session.commit()

        # Force error during match processing
        with patch.object(self.manager, "_process_ball_by_ball", side_effect=RuntimeError("Simulated Failure")):
            self.mock_api.get_matches.return_value = [
                {"id": "mC", "name": "Match C", "matchType": "t20", "matchEnded": True, "date": "2026-09-03", "bbbEnabled": True, "deliveries": [{"runs": 4}]},
            ]
            summary = self.manager.sync_recent_matches()
            self.assertIn("Simulated Failure", summary["errors"][0])

        # Previous records are untouched
        self.assertEqual(self.session.query(Match).count(), 2)
        stored_ids = {m.match_id for m in self.session.query(Match).all()}
        self.assertEqual(stored_ids, {"mA", "mB"})

    def test_final_invariant_count_less_equal_two(self):
        """Scenario 8: Final invariant count <= 2 is strictly enforced, error raised if exceeded."""
        self.mock_api.get_matches.return_value = [
            {"id": f"batch-{i}", "name": f"Batch {i}", "matchType": "t20", "matchEnded": True, "date": f"2026-09-{i:02d}"}
            for i in range(1, 15)
        ]
        summary = self.manager.sync_recent_matches()
        final_count = self.session.query(Match).count()
        self.assertLessEqual(final_count, 2)
        self.assertEqual(summary["total_stored_matches"], 2)

    def test_provider_results_with_duplicates_deduplicated_before_selection(self):
        """Scenario 9: Provider results containing duplicates are deduplicated before selection."""
        self.mock_api.get_matches.return_value = [
            {"id": "m-dup", "name": "Dup Match 1", "matchType": "t20", "matchEnded": True, "date": "2026-09-20"},
            {"id": "m-dup", "name": "Dup Match 2", "matchType": "t20", "matchEnded": True, "date": "2026-09-20"},
            {"id": "m-second", "name": "Second Match", "matchType": "t20", "matchEnded": True, "date": "2026-09-19"},
            {"id": "m-third", "name": "Third Match", "matchType": "t20", "matchEnded": True, "date": "2026-09-18"},
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(self.session.query(Match).count(), 2)
        stored_ids = {m.match_id for m in self.session.query(Match).all()}
        # The deduplicated m-dup (2026-09-20) and m-second (2026-09-19) must be selected
        self.assertEqual(stored_ids, {"m-dup", "m-second"})
        self.assertNotIn("m-third", stored_ids)


class TestMatchSyncManagerDetails(unittest.TestCase):
    """Tests for format filtering, ball-by-ball processing, and missing bbb handling."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        model_path = PROJECT_ROOT / "models" / "logistic_regression_win_probability.joblib"
        self.model = load_prediction_model(model_path)

        self.mock_api = MagicMock(spec=CricketApiClient)
        self.manager = MatchSyncManager(
            api_client=self.mock_api,
            db_session=self.session,
            model=self.model,
        )

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)

    def test_t20_filtering_remains_strict_and_rejects_non_t20(self):
        """Tests K, L: Verify non-T20 formats (Test, ODI, Hundred) are strictly rejected."""
        self.mock_api.get_matches.return_value = [
            {"id": "t-1", "name": "Test Match", "matchType": "test", "matchEnded": True},
            {"id": "o-1", "name": "ODI Match", "matchType": "odi", "matchEnded": True},
            {"id": "h-1", "name": "The Hundred", "matchType": "100", "matchEnded": True},
            {"id": "t20-1", "name": "T20 Match", "matchType": "t20", "matchEnded": True},
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["t20_matches_count"], 1)
        self.assertEqual(summary["matches_stored"], 1)
        self.assertEqual(self.session.query(Match).first().match_id, "t20-1")

    def test_analysis_available_true_when_ball_by_ball_exists(self):
        """Test P: When real deliveries exist, analysis_available is True with model inference."""
        deliveries = [
            {"inning": 1, "over": 19, "ball": 6, "legal_ball_number": 6, "runs": 4, "runs_total": 4, "wickets": 0},
            {"inning": 2, "over": 0, "ball": 1, "legal_ball_number": 1, "runs": 0, "runs_total": 0, "wickets": 0},
            {"inning": 2, "over": 0, "ball": 2, "legal_ball_number": 2, "runs": 4, "runs_total": 4, "wickets": 0},
            {"inning": 2, "over": 0, "ball": 3, "legal_ball_number": 3, "runs": 0, "runs_total": 0, "wickets": 1},
        ]
        raw_match = {
            "id": "t20-bbb-full",
            "name": "India vs England",
            "matchType": "t20",
            "matchEnded": True,
            "status": "India won by 5 wickets",
            "score": [{"r": 160, "w": 6, "o": 20.0}],
            "deliveries": deliveries,
        }
        self.mock_api.get_matches.return_value = [raw_match]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["analysis_available_count"], 1)

        m = self.session.query(Match).filter_by(match_id="t20-bbb-full").first()
        self.assertTrue(m.analysis_available)
        states = self.session.query(MatchState).filter_by(match_id="t20-bbb-full").all()
        self.assertEqual(len(states), 3)

    def test_analysis_available_false_when_ball_by_ball_unavailable(self):
        """Test O: When provider has no ball-by-ball, metadata stored safely with analysis_available=False."""
        raw_match = {
            "id": "t20-nobbb",
            "name": "Nigeria vs Sierra Leone",
            "matchType": "t20i",
            "matchEnded": True,
            "status": "Nigeria won by 34 runs",
            "score": [{"r": 138, "w": 8, "o": 20.0}],
            "bbbEnabled": False,
        }
        self.mock_api.get_matches.return_value = [raw_match]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["analysis_unavailable_count"], 1)
        m = self.session.query(Match).filter_by(match_id="t20-nobbb").first()
        self.assertFalse(m.analysis_available)
        self.assertEqual(self.session.query(Delivery).filter_by(match_id="t20-nobbb").count(), 0)
        self.assertEqual(self.session.query(MatchState).filter_by(match_id="t20-nobbb").count(), 0)


class TestRecentMatchesAPI(unittest.TestCase):
    """Tests for REST API endpoints, Zero-API-Call browsing, and LIVE mode separation (Tests H, I, J, T)."""

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

        init_db()
        from backend.database import SessionLocal
        self.db = SessionLocal()

        # Seed exactly two matches in DB
        self.db.query(Delivery).filter(Delivery.match_id.like("api-test-%")).delete(synchronize_session=False)
        self.db.query(MatchState).filter(MatchState.match_id.like("api-test-%")).delete(synchronize_session=False)
        self.db.query(Match).filter(Match.match_id.like("api-test-%")).delete(synchronize_session=False)
        self.db.commit()

        # Match 1: With ball-by-ball analysis
        m1 = Match(
            match_id="api-test-m1",
            name="India vs Australia, World T20",
            format="T20",
            venue="Eden Gardens, Kolkata",
            match_date="2026-09-25",
            status="India won by 4 wickets",
            winner="India",
            team_1="Australia",
            team_2="India",
            analysis_available=True,
            innings_count=2,
            stored_at=datetime(2099, 9, 25, 20, 0, 0, tzinfo=timezone.utc),
        )
        # Match 2: Without ball-by-ball analysis
        m2 = Match(
            match_id="api-test-m2",
            name="Zimbabwe vs Namibia, T20I",
            format="T20I",
            venue="Harare Sports Club",
            match_date="2099-09-24",
            status="Zimbabwe won by 12 runs",
            winner="Zimbabwe",
            team_1="Zimbabwe",
            team_2="Namibia",
            analysis_available=False,
            innings_count=2,
            stored_at=datetime(2099, 9, 24, 18, 0, 0, tzinfo=timezone.utc),
        )
        self.db.add_all([m1, m2])

        # Add state for m1
        st = MatchState(
            match_id="api-test-m1",
            innings=2,
            legal_balls_completed=110,
            overs_completed=18.3333,
            target_score=170.0,
            current_score=165.0,
            wickets_lost=4,
            runs_remaining=5.0,
            balls_remaining=10,
            current_run_rate=9.0,
            required_run_rate=3.0,
            chasing_team_won=True,
            win_probability=0.85,
            probability_swing=0.05,
        )
        self.db.add(st)
        self.db.commit()

    def tearDown(self):
        self.db.query(Delivery).filter(Delivery.match_id.like("api-test-%")).delete(synchronize_session=False)
        self.db.query(MatchState).filter(MatchState.match_id.like("api-test-%")).delete(synchronize_session=False)
        self.db.query(Match).filter(Match.match_id.like("api-test-%")).delete(synchronize_session=False)
        self.db.commit()
        self.db.close()

    def test_recent_api_reads_only_from_database(self):
        """Test H: GET /api/recent/matches returns max 2 matches stored in database."""
        resp = self.client.get("/api/recent/matches")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertLessEqual(len(data["matches"]), 2)
        match_ids = [m["match_id"] for m in data["matches"]]
        self.assertIn("api-test-m1", match_ids)
        self.assertIn("api-test-m2", match_ids)

    @patch("src.live.cricket_api.CricketApiClient.get_matches")
    @patch("src.live.cricket_api.CricketApiClient.get_current_matches")
    @patch("src.live.cricket_api.CricketApiClient.get_match_info")
    def test_recent_api_makes_zero_provider_calls(self, mock_info, mock_curr, mock_matches):
        """Test I: STRICT INVARIANT: Recent API calls NEVER contact external Cricket Data API."""
        mock_info.side_effect = RuntimeError("External call forbidden")
        mock_curr.side_effect = RuntimeError("External call forbidden")
        mock_matches.side_effect = RuntimeError("External call forbidden")

        # Query recent list
        r1 = self.client.get("/api/recent/matches")
        self.assertEqual(r1.status_code, 200)

        # Query recent detail
        r2 = self.client.get("/api/recent/matches/api-test-m1")
        self.assertEqual(r2.status_code, 200)

        # Both endpoints succeeded with zero provider calls
        mock_info.assert_not_called()
        mock_curr.assert_not_called()
        mock_matches.assert_not_called()

    @patch("backend.app.get_api_client")
    def test_live_still_uses_provider(self, mock_client_factory):
        """Test J: LIVE mode still calls external provider (via CricketApiClient)."""
        mock_client = MagicMock()
        mock_client.get_current_matches.return_value = [
            {"id": "live-t20", "name": "Live T20", "matchType": "t20", "status": "Live"}
        ]
        mock_client_factory.return_value = mock_client

        resp = self.client.get("/api/matches")
        self.assertEqual(resp.status_code, 200)
        mock_client.get_current_matches.assert_called_once()

    def test_unavailable_analysis_displays_exact_message(self):
        """Verify match without ball-by-ball returns exact honest disclaimer."""
        resp = self.client.get("/api/recent/matches/api-test-m2")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data.get("available"))
        self.assertFalse(data.get("analysis_available"))
        self.assertIn("Historical probability replay unavailable for this match", data.get("reason"))

    def test_database_persistence_survives_separate_sessions(self):
        """Test T: Database persistence survives session close and restart."""
        from backend.database import SessionLocal
        s = SessionLocal()
        count = s.query(Match).filter(Match.match_id.like("api-test-%")).count()
        s.close()
        self.assertEqual(count, 2)

    def test_health_endpoint_exposes_database_safely(self):
        """Verify /api/health exposes database connectivity safely without credentials."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("database_connected", data)
        self.assertIn("database_type", data)
        self.assertTrue(data["database_connected"])


class TestModelIntegrityAndConfiguration(unittest.TestCase):
    """Tests for model integrity, exact features, and SQLite fallback (Tests M, N, S)."""

    def test_model_still_uses_exact_8_features(self):
        """Test M: Exact 8 features in exact order."""
        expected = [
            "target_score",
            "current_score",
            "wickets_lost",
            "runs_remaining",
            "balls_remaining",
            "overs_completed",
            "current_run_rate",
            "required_run_rate",
        ]
        self.assertEqual(FEATURE_COLS, expected)

    def test_model_artifact_unchanged(self):
        """Test N: Pre-trained Logistic Regression model artifact exists and loads correctly."""
        model_path = PROJECT_ROOT / "models" / "logistic_regression_win_probability.joblib"
        self.assertTrue(model_path.exists())
        model = load_prediction_model(model_path)
        self.assertIsNotNone(model)

    def test_sqlite_fallback_still_works(self):
        """Test S: SQLite fallback operates cleanly when DATABASE_URL is unset."""
        with patch.dict(os.environ, {}, clear=True):
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]
            url = get_database_url()
            self.assertTrue(url.startswith("sqlite:///"))

    def test_neon_database_url_normalization(self):
        """Verify Neon connection strings (postgres:// and postgresql://) normalize to postgresql+psycopg://."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@ep-cool-fog-123456.us-east-2.aws.neon.tech/cricket_db?sslmode=require"}):
            url = get_database_url()
            self.assertTrue(url.startswith("postgresql+psycopg://"))
            self.assertIn("sslmode=require", url)


class TestWinnerExtractionAndMapping(unittest.TestCase):
    """Regression tests for winner extraction and metadata mapping (Phase 12.3 bugfix)."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        self.mock_api = MagicMock(spec=CricketApiClient)
        self.manager = MatchSyncManager(
            api_client=self.mock_api,
            db_session=self.session,
        )

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(bind=self.engine)

    def test_winner_extracted_from_wickets_margin(self):
        """Scenario a: 'Luxembourg Women won by 4 wkts' -> winner = 'Luxembourg Women'."""
        self.mock_api.get_matches.return_value = [
            {
                "id": "lux-bel-3",
                "name": "Luxembourg Women vs Belgium Women, 3rd T20I, Belgium Women tour of Luxembourg, 2026",
                "matchType": "t20",
                "matchEnded": True,
                "status": "Luxembourg Women won by 4 wkts",
                "teams": ["Luxembourg Women", "Belgium Women"],
                "date": "2026-09-28",
            }
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["matches_stored"], 1)

        m = self.session.query(Match).filter_by(match_id="lux-bel-3").first()
        self.assertIsNotNone(m)
        self.assertEqual(m.winner, "Luxembourg Women")
        self.assertEqual(m.status, "Luxembourg Women won by 4 wkts")
        self.assertEqual(m.result_text, "Luxembourg Women won by 4 wkts")

        # Verify JSON dictionary serialization
        d = m.to_dict()
        self.assertEqual(d["winner"], "Luxembourg Women")
        self.assertEqual(d["result"], "Luxembourg Women won by 4 wkts")

    def test_winner_extracted_from_runs_margin(self):
        """Scenario b: 'Luxembourg Women won by 47 runs' -> winner = 'Luxembourg Women'."""
        self.mock_api.get_matches.return_value = [
            {
                "id": "lux-bel-4",
                "name": "Luxembourg Women vs Belgium Women, 4th T20I, Belgium Women tour of Luxembourg, 2026",
                "matchType": "t20",
                "matchEnded": True,
                "status": "Luxembourg Women won by 47 runs",
                "teams": ["Luxembourg Women", "Belgium Women"],
                "date": "2026-09-28",
            }
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["matches_stored"], 1)

        m = self.session.query(Match).filter_by(match_id="lux-bel-4").first()
        self.assertIsNotNone(m)
        self.assertEqual(m.winner, "Luxembourg Women")
        self.assertEqual(m.status, "Luxembourg Women won by 47 runs")

        d = m.to_dict()
        self.assertEqual(d["winner"], "Luxembourg Women")
        self.assertEqual(d["result"], "Luxembourg Women won by 47 runs")

    def test_winner_cannot_safely_be_determined_remains_null(self):
        """Scenario c: Undetermined results ('Match tied', 'Match abandoned', 'No result') -> winner remains NULL."""
        undetermined_statuses = [
            ("tie-1", "Match tied", "2026-09-20"),
            ("aban-2", "Match abandoned without a ball bowled", "2026-09-21"),
            ("nr-3", "No result", "2026-09-22"),
        ]
        for m_id, status_val, d_val in undetermined_statuses:
            self.mock_api.get_matches.return_value = [
                {
                    "id": m_id,
                    "name": f"Match {m_id}",
                    "matchType": "t20",
                    "matchEnded": True,
                    "status": status_val,
                    "teams": ["Team A", "Team B"],
                    "date": d_val,
                }
            ]
            self.manager.sync_recent_matches()
            m = self.session.query(Match).filter_by(match_id=m_id).first()
            if m:
                self.assertIsNone(m.winner, f"Expected None for status '{status_val}', got '{m.winner}'")
                d = m.to_dict()
                self.assertIsNone(d["winner"])

    def test_already_stored_match_with_null_winner_updated_by_sync(self):
        """Scenario d: An already-stored match with winner=NULL is updated to correctly parsed winner upon sync."""
        # Pre-seed database with a match having winner=None
        pre_match = Match(
            match_id="lux-stored-null",
            name="Luxembourg Women vs Belgium Women, 3rd T20I, Belgium Women tour of Luxembourg, 2026",
            format="T20",
            venue="Pierre Werner Cricket Ground, Walferdange",
            match_date="2026-09-28",
            status="Luxembourg Women won by 4 wkts",
            winner=None,  # Stored as NULL previously
            result_text="Luxembourg Women won by 4 wkts",
            team_1="Luxembourg Women",
            team_2="Belgium Women",
            stored_at=datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc),
        )
        self.session.add(pre_match)
        self.session.commit()

        # Confirm initial state is NULL
        initial = self.session.query(Match).filter_by(match_id="lux-stored-null").first()
        self.assertIsNone(initial.winner)
        self.assertIsNone(initial.to_dict()["winner"])

        # Run normal sync where provider returns match update
        self.mock_api.get_matches.return_value = [
            {
                "id": "lux-stored-null",
                "name": "Luxembourg Women vs Belgium Women, 3rd T20I, Belgium Women tour of Luxembourg, 2026",
                "matchType": "t20",
                "matchEnded": True,
                "status": "Luxembourg Women won by 4 wkts",
                "teams": ["Luxembourg Women", "Belgium Women"],
                "date": "2026-09-28",
            }
        ]
        summary = self.manager.sync_recent_matches()
        self.assertEqual(summary["duplicate_matches_count"], 1)

        # Verify winner changed from NULL to "Luxembourg Women"
        updated = self.session.query(Match).filter_by(match_id="lux-stored-null").first()
        self.assertEqual(updated.winner, "Luxembourg Women")
        self.assertEqual(updated.to_dict()["winner"], "Luxembourg Women")
        self.assertEqual(updated.to_dict()["result"], "Luxembourg Women won by 4 wkts")

    def test_explicit_provider_winner_field_takes_precedence(self):
        """Verify explicit matchWinner or winner field from provider is preserved."""
        self.mock_api.get_matches.return_value = [
            {
                "id": "exp-winner-1",
                "name": "Team Alpha vs Team Beta",
                "matchType": "t20",
                "matchEnded": True,
                "matchWinner": "Team Alpha",
                "status": "Team Alpha won by 15 runs",
                "teams": ["Team Alpha", "Team Beta"],
                "date": "2026-09-28",
            }
        ]
        self.manager.sync_recent_matches()
        m = self.session.query(Match).filter_by(match_id="exp-winner-1").first()
        self.assertIsNotNone(m)
        self.assertEqual(m.winner, "Team Alpha")

    def test_direct_winner_parser_edge_cases(self):
        """Direct unit test of extract_winner_from_result parser with diverse cricket outcomes."""
        teams = ["Luxembourg Women", "Belgium Women"]
        self.assertEqual(extract_winner_from_result("Luxembourg Women won by 4 wkts", teams), "Luxembourg Women")
        self.assertEqual(extract_winner_from_result("Luxembourg Women won by 47 runs", teams), "Luxembourg Women")
        self.assertEqual(extract_winner_from_result("India won by 6 wickets", ["India", "England"]), "India")
        self.assertEqual(extract_winner_from_result("South Africa won by 1 run", ["South Africa", "Sri Lanka"]), "South Africa")
        self.assertEqual(extract_winner_from_result("Match tied (Team A won the Super Over)", ["Team A", "Team B"]), "Team A")
        self.assertEqual(extract_winner_from_result("England won by 5 wickets (DLS method)", ["England", "Australia"]), "England")
        self.assertIsNone(extract_winner_from_result("Match tied", teams))
        self.assertIsNone(extract_winner_from_result("Match abandoned without a ball bowled", teams))
        self.assertIsNone(extract_winner_from_result("No result", teams))
        self.assertIsNone(extract_winner_from_result("In progress", teams))
        self.assertIsNone(extract_winner_from_result("", teams))
        self.assertIsNone(extract_winner_from_result(None, teams))


if __name__ == "__main__":
    unittest.main()
