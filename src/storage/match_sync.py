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
        init_db()

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

    def apply_two_match_retention(self, session) -> int:
        """
        Enforces Phase 12.3 invariant: MAXIMUM 2 stored matches.
        Orders stored matches by stored_at descending, match_date descending.
        Retains the 2 newest matches and deletes all older matches within the transaction.
        Returns the count of deleted matches.
        """
        all_stored = (
            session.query(Match)
            .order_by(Match.stored_at.desc(), Match.match_date.desc())
            .all()
        )
        if len(all_stored) <= 2:
            return 0

        to_delete = all_stored[2:]
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

        # 1. Query recent matches from API
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
            for raw_m in raw_matches[:limit]:
                # 2. Strict T20/T20I Filter
                if not is_t20_match(raw_m):
                    continue
                summary["t20_matches_count"] += 1

                # 3. Identify completed matches
                is_ended = bool(raw_m.get("matchEnded", False))
                status_str = str(raw_m.get("status") or "").lower()
                is_completed_by_status = any(
                    kw in status_str for kw in ["won by", "tied", "match tied", "abandoned", "no result"]
                )

                if not (is_ended or is_completed_by_status):
                    # In-progress or scheduled match, not completed
                    continue
                summary["completed_matches_count"] += 1

                match_id = str(raw_m.get("id") or "").strip()
                if not match_id:
                    continue

                # Parse match metadata
                match_name = str(raw_m.get("name") or "Unknown T20 Match").strip()
                match_format = str(raw_m.get("matchType") or "t20").upper()
                series_name = str(raw_m.get("series_name") or raw_m.get("series_id") or "").strip()
                venue = str(raw_m.get("venue") or "").strip()
                city = str(raw_m.get("city") or "").strip()
                match_date = str(raw_m.get("date") or "").strip()
                date_time = str(raw_m.get("dateTimeGMT") or raw_m.get("dateTime") or raw_m.get("date") or "").strip()
                status = str(raw_m.get("status") or "").strip()
                winner = str(raw_m.get("matchWinner") or "").strip()

                teams = [str(t).strip() for t in raw_m.get("teams", []) if t]
                team_1 = teams[0] if len(teams) > 0 else None
                team_2 = teams[1] if len(teams) > 1 else None

                scores = raw_m.get("score", [])
                innings_count = len(scores) if isinstance(scores, list) else 0

                # Check if ball-by-ball data is available from provider
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

                # 4. Check if match already exists in database (Deduplication)
                existing_match = session.query(Match).filter_by(match_id=match_id).first()
                if existing_match:
                    existing_match.status = status or existing_match.status
                    existing_match.winner = winner or existing_match.winner
                    existing_match.result_text = status or existing_match.result_text
                    existing_match.updated_at = datetime.now(timezone.utc)
                    if not existing_match.analysis_available and has_bbb:
                        deliv_count, state_count = self._process_ball_by_ball(
                            session=session,
                            match_record=existing_match,
                            deliveries_data=deliveries_list,
                            raw_scores=scores,
                        )
                        if state_count > 0:
                            existing_match.analysis_available = True
                            summary["analysis_available_count"] += 1
                            summary["deliveries_stored"] += deliv_count
                            summary["states_stored"] += state_count
                    summary["duplicate_matches_count"] += 1
                    summary["matches_skipped"] += 1
                    deleted = self.apply_two_match_retention(session)
                    summary["deleted_old_matches"] += deleted
                    session.commit()
                    continue

                # 5. Process new completed match
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
                    winner=winner or None,
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

                # Enforce two-match retention transactionally
                deleted = self.apply_two_match_retention(session)
                summary["deleted_old_matches"] += deleted

                session.commit()
                summary["matches_stored"] += 1
                logger.info(
                    f"Stored match: {match_name} (analysis_available={match_record.analysis_available})"
                )

            summary["total_stored_matches"] = session.query(Match).count()

        except Exception as e:
            session.rollback()
            logger.error(f"Error during database ingestion: {e}")
            summary["errors"].append(str(e))
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
