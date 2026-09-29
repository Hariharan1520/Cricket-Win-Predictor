"""
Controlled match synchronization and ingestion module for persistent storage (Phase 12).

Synchronizes completed T20/T20I matches from the Cricket Data API into PostgreSQL/SQLite.
Performs ball-by-ball analysis, evaluates the frozen Logistic Regression model,
calculates Win Probability Swings, and stores match states for instant historical replay.

Usage:
    python -m src.storage.match_sync --recent
    python -m src.storage.match_sync --recent --limit 25
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import Delivery, Match, MatchState, SessionLocal, init_db
from src.live.cricket_api import (
    CricketApiClient,
    CricketApiError,
    CricketApiAuthError,
    CricketApiRateLimitError,
)
from src.live.live_features import (
    FEATURE_COLS,
    is_t20_match,
    normalize_match_dict,
    parse_cricket_overs,
)
from src.live.live_match import load_prediction_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("match_sync")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_win_probability.joblib"


def extract_winner_from_result(status_str: Optional[str], teams: Optional[List[str]] = None) -> Optional[str]:
    """
    Derives the winning team name from a match result/status string.
    E.g. 'Luxembourg Women won by 4 wkts' -> 'Luxembourg Women'
         'Luxembourg Women won by 47 runs' -> 'Luxembourg Women'
         'Match tied (Team A won the Super Over)' -> 'Team A'
         'Match tied' -> None
         'Match abandoned' -> None
         'No result' -> None
    """
    if not status_str or not isinstance(status_str, str):
        return None
    s = status_str.strip()
    if not s:
        return None

    s_lower = s.lower()
    # Non-decisive or non-winner match conclusions
    if any(k in s_lower for k in ["no result", "abandoned", "drawn", "cancelled", "rain stopped", "suspended"]):
        return None
    if "tied" in s_lower and "won" not in s_lower:
        return None

    candidate = None
    # 1. Super Over / parenthesis decider: "Match tied (Team A won ...)"
    paren_match = re.search(r"\(\s*([^\)]+?)\s+won\b", s, re.IGNORECASE)
    if paren_match:
        candidate = paren_match.group(1).strip()
    else:
        # 2. Standard pattern: "<Team Name> won ..."
        m = re.match(r"^(.+?)\s+won(?:\s+by|\s+the|\s+on|\b)", s, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()

    if not candidate or candidate.lower() in ["match", "the match", "no"]:
        return None

    # If known teams list provided, match casing and verify
    if teams:
        for t in teams:
            t_str = str(t).strip()
            if t_str and candidate.lower() == t_str.lower():
                return t_str

    return candidate


def extract_match_winner(raw_m: Dict[str, Any], teams: Optional[List[str]] = None) -> Optional[str]:
    """
    Extracts or derives the winning team for a match.
    1. Checks explicit provider fields: 'matchWinner' or 'winner'.
    2. If missing or empty, derives it from the 'status' or 'result' string.
    3. Returns None if winner cannot safely be determined.
    """
    # 1. Check explicit fields
    for key in ("matchWinner", "winner"):
        val = raw_m.get(key)
        if val and str(val).strip():
            explicit_str = str(val).strip()
            if teams:
                for t in teams:
                    if explicit_str.lower() == str(t).strip().lower():
                        return str(t).strip()
            return explicit_str

    # 2. Derive from status / result text
    status_str = str(raw_m.get("status") or raw_m.get("result") or "").strip()
    return extract_winner_from_result(status_str, teams)


class MatchSyncManager:
    """
    Manages synchronization of completed T20 matches into persistent storage.
    Enforces strict T20 filtering, deduplication, and zero synthetic data insertion.
    """

    def __init__(
        self,
        api_client: Optional[CricketApiClient] = None,
        db_session=None,
        model=None,
    ):
        self._api_client = api_client
        self._db_session = db_session
        self.model = model

        # Ensure database tables exist
        target_eng = None
        if self._db_session is not None:
            try:
                target_eng = self._db_session.get_bind()
            except Exception:
                pass
        try:
            init_db(target_engine=target_eng)
        except Exception as e:
            logger.warning(f"Database initialization note: {e}")

        if self.model is None and DEFAULT_MODEL_PATH.exists():
            try:
                self.model = load_prediction_model(DEFAULT_MODEL_PATH)
                logger.info("Loaded frozen Logistic Regression model for historical analysis.")
            except Exception as e:
                logger.warning(f"Could not load prediction model from {DEFAULT_MODEL_PATH}: {e}")

    @property
    def api_client(self) -> CricketApiClient:
        if self._api_client is None:
            self._api_client = CricketApiClient()
        return self._api_client

    def get_session(self):
        if self._db_session is not None:
            return self._db_session
        return SessionLocal()

    @staticmethod
    def _parse_date_to_datetime(d_val: Any) -> datetime:
        """Parses a date or ISO string into a timezone-aware UTC datetime."""
        if not d_val:
            return datetime.min.replace(tzinfo=timezone.utc)
        raw = str(d_val).strip()
        if not raw:
            return datetime.min.replace(tzinfo=timezone.utc)

        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass

        return datetime.min.replace(tzinfo=timezone.utc)

    def apply_two_match_retention(self, session, retain_ids: Optional[set] = None) -> int:
        """
        Enforces Phase 12.3 invariant: MAXIMUM 2 stored matches.
        If retain_ids is provided, deletes all stored matches not in retain_ids.
        Otherwise, orders stored matches by recency and deletes older matches beyond the top 2.
        Returns the count of deleted matches.
        """
        all_stored = session.query(Match).all()
        if not all_stored:
            return 0

        if retain_ids is not None:
            to_delete = [m for m in all_stored if m.match_id not in retain_ids]
        else:
            if len(all_stored) <= 2:
                return 0
            sorted_stored = sorted(
                all_stored,
                key=lambda m: (
                    self._parse_date_to_datetime(m.date_time or m.match_date),
                    m.stored_at.replace(tzinfo=timezone.utc) if (m.stored_at and m.stored_at.tzinfo is None) else (m.stored_at or datetime.min.replace(tzinfo=timezone.utc)),
                    m.match_id,
                ),
                reverse=True,
            )
            to_delete = sorted_stored[2:]

        deleted_count = 0
        for old_m in to_delete:
            logger.info(f"Two-match retention: deleting older match {old_m.match_id} ({old_m.name})")
            session.delete(old_m)
            deleted_count += 1

        return deleted_count

    def sync_recent_matches(self, limit: int = 25, offset: int = 0) -> Dict[str, Any]:
        """
        Fetches completed T20/T20I fixtures, filters, deduplicates, and persists.
        Enforces strict Two-Match Retention: only the 2 most recent completed matches remain stored.
        Never inserts synthetic matches.
        """
        summary = {
            "provider_api_calls": 0,
            "matches_discovered": 0,
            "t20_matches_count": 0,
            "completed_matches_count": 0,
            "matches_stored": 0,
            "matches_skipped": 0,
            "duplicate_matches_count": 0,
            "deleted_old_matches": 0,
            "total_stored_matches": 0,
            "analysis_available_count": 0,
            "analysis_unavailable_count": 0,
            "deliveries_stored": 0,
            "states_stored": 0,
            "errors": [],
            "quota_events": 0,
        }

        # 1. Query Cricket Data API once
        try:
            logger.info(f"Querying Cricket Data API /matches (offset={offset})...")
            raw_matches = self.api_client.get_matches(offset=offset)
            summary["provider_api_calls"] += 1
            summary["matches_discovered"] = len(raw_matches)
        except CricketApiRateLimitError as e:
            logger.warning(f"API Rate Limit reached during sync: {e}")
            summary["quota_events"] += 1
            summary["errors"].append(f"Rate limit exceeded: {e}")
            return summary
        except CricketApiError as e:
            logger.error(f"Provider API error during sync: {e}")
            summary["errors"].append(f"Provider error: {e}")
            return summary
        except Exception as e:
            logger.error(f"Unexpected error querying provider: {e}")
            summary["errors"].append(f"Unexpected error: {e}")
            return summary

        session = self.get_session()
        close_session = self._db_session is None

        try:
            # 2. Filter strictly to completed T20/T20I matches
            # 3. Deduplicate provider results by stable match_id
            valid_provider_by_id: Dict[str, Dict[str, Any]] = {}
            for raw_m in raw_matches[:limit]:
                if not is_t20_match(raw_m):
                    continue
                summary["t20_matches_count"] += 1

                is_ended = bool(raw_m.get("matchEnded", False))
                status_str = str(raw_m.get("status") or "").lower()
                is_completed_by_status = any(
                    kw in status_str for kw in ["won by", "tied", "match tied", "abandoned", "no result"]
                )

                if not (is_ended or is_completed_by_status):
                    continue
                summary["completed_matches_count"] += 1

                match_id = str(raw_m.get("id") or "").strip()
                if not match_id:
                    continue

                if match_id not in valid_provider_by_id:
                    valid_provider_by_id[match_id] = raw_m

            # If zero valid matches discovered from provider:
            # Existing database should remain unchanged.
            if not valid_provider_by_id:
                logger.info("Zero valid completed T20 matches discovered from provider. Database remains unchanged.")
                summary["total_stored_matches"] = session.query(Match).count()
                return summary

            # Query existing matches from database
            existing_db_matches = session.query(Match).all()
            existing_by_id = {m.match_id: m for m in existing_db_matches}

            # 4. Sort valid matches by actual recency:
            # - match_date DESC
            # - stored_at DESC where applicable
            candidates: Dict[str, Tuple[datetime, datetime, Optional[Dict[str, Any]], Optional[Match]]] = {}

            for match_id, raw_m in valid_provider_by_id.items():
                m_dt = self._parse_date_to_datetime(
                    raw_m.get("dateTimeGMT") or raw_m.get("dateTime") or raw_m.get("date")
                )
                existing = existing_by_id.get(match_id)
                if existing and existing.stored_at:
                    s_at = existing.stored_at
                else:
                    s_at = datetime.now(timezone.utc)
                if s_at.tzinfo is None:
                    s_at = s_at.replace(tzinfo=timezone.utc)
                candidates[match_id] = (m_dt, s_at, raw_m, existing)

            for match_id, db_m in existing_by_id.items():
                if match_id not in candidates:
                    m_dt = self._parse_date_to_datetime(db_m.date_time or db_m.match_date)
                    s_at = db_m.stored_at if db_m.stored_at else datetime.min.replace(tzinfo=timezone.utc)
                    if s_at.tzinfo is None:
                        s_at = s_at.replace(tzinfo=timezone.utc)
                    candidates[match_id] = (m_dt, s_at, None, db_m)

            sorted_candidates = sorted(
                candidates.items(),
                key=lambda item: (item[1][0], item[1][1], item[0]),
                reverse=True,
            )

            # 5. Select the newest 2 valid matches
            selected_candidates = sorted_candidates[:2]
            selected_ids = {item[0] for item in selected_candidates}

            # 6. Begin ONE database transaction
            # (session is already bound to a transaction in SQLAlchemy)

            # 7. Upsert/update ONLY those selected newest 2 matches
            for match_id, (m_dt, s_at, raw_m, existing_record) in selected_candidates:
                if raw_m is not None:
                    match_name = str(raw_m.get("name") or "Unknown T20 Match").strip()
                    match_format = str(raw_m.get("matchType") or "t20").upper()
                    series_name = str(raw_m.get("series_name") or raw_m.get("series_id") or "").strip()
                    venue = str(raw_m.get("venue") or "").strip()
                    city = str(raw_m.get("city") or "").strip()
                    match_date = str(raw_m.get("date") or "").strip()
                    date_time = str(raw_m.get("dateTimeGMT") or raw_m.get("dateTime") or raw_m.get("date") or "").strip()
                    status = str(raw_m.get("status") or "").strip()

                    teams = [str(t).strip() for t in raw_m.get("teams", []) if t]
                    if not teams and existing_record is not None:
                        teams = [t for t in [existing_record.team_1, existing_record.team_2] if t]
                    team_1 = teams[0] if len(teams) > 0 else (existing_record.team_1 if existing_record else None)
                    team_2 = teams[1] if len(teams) > 1 else (existing_record.team_2 if existing_record else None)

                    winner = extract_match_winner(raw_m, teams=teams)

                    scores = raw_m.get("score", [])
                    innings_count = len(scores) if isinstance(scores, list) else 0

                    bbb_enabled = bool(raw_m.get("bbbEnabled", False))
                    deliveries_list = raw_m.get("deliveries") or raw_m.get("bbb")

                    if bbb_enabled and not deliveries_list:
                        try:
                            logger.info(f"Fetching ball-by-ball match detail for match {match_id}...")
                            m_detail = self.api_client.get_match_info(match_id)
                            summary["provider_api_calls"] += 1
                            deliveries_list = m_detail.get("deliveries") or m_detail.get("bbb")
                        except Exception as e:
                            logger.warning(f"Failed to fetch detail for {match_id}: {e}")

                    has_bbb = bool(deliveries_list and isinstance(deliveries_list, list) and len(deliveries_list) > 0)

                    if existing_record is not None:
                        # Existing match update
                        existing_record.status = status or existing_record.status
                        existing_record.winner = winner or existing_record.winner
                        existing_record.result_text = status or existing_record.result_text
                        existing_record.team_1 = team_1 or existing_record.team_1
                        existing_record.team_2 = team_2 or existing_record.team_2
                        existing_record.updated_at = datetime.now(timezone.utc)
                        if not existing_record.analysis_available and has_bbb:
                            deliv_count, state_count = self._process_ball_by_ball(
                                session=session,
                                match_record=existing_record,
                                deliveries_data=deliveries_list,
                                raw_scores=scores,
                            )
                            if state_count > 0:
                                existing_record.analysis_available = True
                                summary["analysis_available_count"] += 1
                                summary["deliveries_stored"] += deliv_count
                                summary["states_stored"] += state_count
                        summary["duplicate_matches_count"] += 1
                        summary["matches_skipped"] += 1
                    else:
                        # New match insert
                        match_record = Match(
                            match_id=match_id,
                            name=match_name,
                            format=match_format,
                            series_name=series_name or None,
                            venue=venue or None,
                            city=city or None,
                            match_date=match_date or None,
                            date_time=date_time or None,
                            completed_at=date_time or match_date or None,
                            status=status or None,
                            winner=winner,
                            result_text=status or None,
                            team_1=team_1,
                            team_2=team_2,
                            source="cricket_api",
                            source_updated_at=datetime.now(timezone.utc),
                            stored_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc),
                            analysis_available=False,
                            innings_count=innings_count,
                        )
                        session.add(match_record)

                        if has_bbb:
                            deliv_count, state_count = self._process_ball_by_ball(
                                session=session,
                                match_record=match_record,
                                deliveries_data=deliveries_list,
                                raw_scores=scores,
                            )
                            if state_count > 0:
                                match_record.analysis_available = True
                                summary["analysis_available_count"] += 1
                                summary["deliveries_stored"] += deliv_count
                                summary["states_stored"] += state_count
                            else:
                                summary["analysis_unavailable_count"] += 1
                        else:
                            summary["analysis_unavailable_count"] += 1

                        summary["matches_stored"] += 1
                        logger.info(
                            f"Stored match: {match_name} (analysis_available={match_record.analysis_available})"
                        )
                else:
                    # Match was already in DB and remains selected as one of the 2 newest
                    # Backfill winner if missing and result_text/status exists
                    if existing_record and not existing_record.winner:
                        teams = [t for t in [existing_record.team_1, existing_record.team_2] if t]
                        derived_winner = extract_winner_from_result(
                            existing_record.result_text or existing_record.status or "",
                            teams=teams,
                        )
                        if derived_winner:
                            existing_record.winner = derived_winner
                            existing_record.updated_at = datetime.now(timezone.utc)

            # 8. Delete every stored Match whose match_id is NOT in the selected newest 2
            to_delete = session.query(Match).filter(~Match.match_id.in_(selected_ids)).all()
            for old_m in to_delete:
                logger.info(f"Two-match retention: deleting older match {old_m.match_id} ({old_m.name})")
                session.delete(old_m)
                summary["deleted_old_matches"] += 1

            # 9. Commit once
            session.commit()

            # 11. At the end, query the database again and verify count <= 2
            final_count = session.query(Match).count()
            summary["total_stored_matches"] = final_count

            # 12. If count > 2, raise an error rather than silently reporting success
            if final_count > 2:
                raise RuntimeError(
                    f"Retention invariant violation: database has {final_count} stored matches (maximum 2 allowed)."
                )

        except Exception as e:
            session.rollback()
            logger.error(f"Error during database ingestion: {e}")
            summary["errors"].append(str(e))
            if "Retention invariant violation" in str(e):
                raise
        finally:
            if close_session:
                session.close()

        return summary

    def _process_ball_by_ball(
        self,
        session,
        match_record: Match,
        deliveries_data: List[Dict[str, Any]],
        raw_scores: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Processes real deliveries, computes the exact 8 features per second-innings delivery,
        runs the frozen Logistic Regression model, computes swings, and persists records.
        """
        deliv_count = 0
        state_count = 0

        # Calculate target score from 1st innings if available
        target_score = 0.0
        if raw_scores and len(raw_scores) >= 1:
            inn1_runs = int(raw_scores[0].get("r", 0) or 0)
            target_score = float(inn1_runs + 1)

        # Store deliveries
        deliveries_to_add = []
        for d in deliveries_data:
            inn = int(d.get("inning", d.get("innings", 1)) or 1)
            over_num = int(d.get("over", d.get("over_number", 0)) or 0)
            ball_num = int(d.get("ball", d.get("ball_number", 1)) or 1)
            legal_ball = int(d.get("legal_ball_number", ball_num) or ball_num)

            deliv = Delivery(
                match_id=match_record.match_id,
                innings=inn,
                over_number=over_num,
                ball_number=ball_num,
                legal_ball_number=legal_ball,
                batter=str(d.get("batter", "")) or None,
                bowler=str(d.get("bowler", "")) or None,
                non_striker=str(d.get("non_striker", "")) or None,
                runs_batter=int(d.get("runs_batter", 0) or 0),
                runs_total=int(d.get("runs_total", d.get("runs", 0)) or 0),
                extras=int(d.get("extras", 0) or 0),
                wickets=int(d.get("wickets", 0) or 0),
                raw_info=json.dumps(d) if isinstance(d, dict) else str(d),
            )
            deliveries_to_add.append(deliv)
            deliv_count += 1

        session.add_all(deliveries_to_add)

        # Build second-innings match states for historical replay
        inn2_deliveries = [d for d in deliveries_data if int(d.get("inning", d.get("innings", 1)) or 1) == 2]
        if not inn2_deliveries or target_score <= 0 or self.model is None:
            return deliv_count, 0

        cur_score = 0.0
        wickets_lost = 0
        legal_balls_completed = 0
        prev_prob = 0.5  # Starting baseline

        states_by_ball = {}
        for d in inn2_deliveries:
            runs_total = int(d.get("runs_total", d.get("runs", 0)) or 0)
            wickets = int(d.get("wickets", 0) or 0)
            is_extra_noball = bool(d.get("noball") or d.get("is_noball", False))
            is_extra_wide = bool(d.get("wide") or d.get("is_wide", False))

            cur_score += runs_total
            wickets_lost = min(10, wickets_lost + wickets)

            if not (is_extra_wide or is_extra_noball):
                legal_balls_completed = min(120, legal_balls_completed + 1)

            balls_remaining = max(0, 120 - legal_balls_completed)
            overs_completed = round(legal_balls_completed / 6.0, 4)
            runs_remaining = max(0.0, target_score - cur_score)
            current_rr = round(cur_score / overs_completed, 4) if overs_completed > 0 else 0.0
            required_rr = (
                round(runs_remaining / (balls_remaining / 6.0), 4)
                if balls_remaining > 0 and runs_remaining > 0
                else 0.0
            )

            # Terminal states
            if cur_score >= target_score:
                win_prob = 1.0
            elif wickets_lost >= 10 or balls_remaining <= 0:
                win_prob = 0.0
            else:
                features_df = pd.DataFrame(
                    [
                        {
                            "target_score": target_score,
                            "current_score": cur_score,
                            "wickets_lost": wickets_lost,
                            "runs_remaining": runs_remaining,
                            "balls_remaining": balls_remaining,
                            "overs_completed": overs_completed,
                            "current_run_rate": current_rr,
                            "required_run_rate": required_rr,
                        }
                    ]
                )[FEATURE_COLS]
                proba = self.model.predict_proba(features_df)
                win_prob = float(proba[0][1])

            swing = round(win_prob - prev_prob, 4)
            prev_prob = win_prob

            state_record = MatchState(
                match_id=match_record.match_id,
                innings=2,
                legal_balls_completed=legal_balls_completed,
                overs_completed=overs_completed,
                target_score=target_score,
                current_score=cur_score,
                wickets_lost=wickets_lost,
                runs_remaining=runs_remaining,
                balls_remaining=balls_remaining,
                current_run_rate=current_rr,
                required_run_rate=required_rr,
                chasing_team_won=match_record.winner == match_record.team_2 if match_record.winner else None,
                win_probability=win_prob,
                probability_swing=swing,
            )
            states_by_ball[legal_balls_completed] = state_record

        states_to_add = list(states_by_ball.values())
        session.add_all(states_to_add)
        state_count = len(states_to_add)
        return deliv_count, state_count


def parse_args():
    parser = argparse.ArgumentParser(description="Synchronize recent completed T20 matches into persistent database.")
    parser.add_argument("--recent", action="store_true", help="Sync recent completed T20 matches from Cricket Data API")
    parser.add_argument("--limit", type=int, default=25, help="Maximum matches to process")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset for provider API")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.recent:
        print("Usage: python -m src.storage.match_sync --recent [--limit N] [--offset N]")
        sys.exit(0)

    manager = MatchSyncManager()
    print("\n--- STARTING CONTROLLED RECENT T20 MATCH SYNC ---")
    summary = manager.sync_recent_matches(limit=args.limit, offset=args.offset)
    print("\n--- SYNC COMPLETED ---")
    print(f"Provider matches checked:         {summary['matches_discovered']}")
    print(f"Valid completed T20/T20I matches: {summary['completed_matches_count']}")
    print(f"Inserted:                         {summary['matches_stored']}")
    print(f"Updated / Skipped duplicates:     {summary['duplicate_matches_count']}")
    print(f"Stored matches (max 2):           {summary['total_stored_matches']}")
    print(f"Deleted old matches:              {summary['deleted_old_matches']}")
    print(f"Analysis Available:               {summary['analysis_available_count']}")
    print(f"Analysis Unavailable:             {summary['analysis_unavailable_count']}")
    print(f"Deliveries Stored:                {summary['deliveries_stored']}")
    print(f"States Stored:                    {summary['states_stored']}")
    print(f"Provider API Calls Used:          {summary['provider_api_calls']}")
    if summary["errors"]:
        print(f"Errors ({len(summary['errors'])}):")
        for err in summary["errors"]:
            print(f"  - {err}")
    print("----------------------\n")


if __name__ == "__main__":
    main()
