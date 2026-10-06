"""
Flask Backend REST API for T20 Cricket Win Predictor Dashboard.

Connects the React frontend to:
- Cricket Data API live feeds (T20 only)
- Frozen Logistic Regression Win-Probability Model (Phase 3)
- What-If Next-Over Simulator (Phase 6)
- Live State Tracker and Swing Analysis (Phases 5 & 7)
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env — but never override an already-set DATABASE_URL (e.g. from tests)
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file, override=False)
else:
    load_dotenv(override=False)

from src.live.cricket_api import (
    CricketApiClient,
    CricketApiError,
    CricketApiAuthError,
    CricketApiRateLimitError,
)
from src.live.live_features import (
    FEATURE_COLS,
    LiveMatchFeatures,
    NormalizedMatch,
    extract_match_state_features,
    filter_t20_matches,
    is_completed_match,
    is_t20_match,
    normalize_match_dict,
    normalize_match_status,
)
from src.live.live_match import (
    LivePredictionResult,
    LiveStateTracker,
    load_prediction_model,
    predict_live_state,
)
from src.simulation.next_over_simulator import (
    BASELINE_SCENARIOS,
    simulate_next_over,
)
from backend.database import Delivery, Match, MatchState, check_db_connection, get_db_session, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard_backend")

# Apply safe schema migrations before serving requests.
init_db()

app = Flask(__name__)

# Configure production-ready CORS origin filtering
frontend_origin_env = os.getenv("FRONTEND_ORIGIN", "").strip()
if frontend_origin_env and frontend_origin_env != "*":
    allowed_origins = [o.strip() for o in frontend_origin_env.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    logger.info(f"CORS initialized with restricted origins: {allowed_origins}")
else:
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    logger.info("CORS initialized with permissive origins (*). Configure FRONTEND_ORIGIN for production.")

# Load model once on startup
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_win_probability.joblib"
try:
    model = load_prediction_model(MODEL_PATH)
    logger.info("Loaded Logistic Regression win-probability model successfully.")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None

# Initialize state trackers per match
match_trackers: Dict[str, LiveStateTracker] = {}


def get_or_create_tracker(match_id: str) -> LiveStateTracker:
    if match_id not in match_trackers:
        match_trackers[match_id] = LiveStateTracker()
    return match_trackers[match_id]


def get_api_client() -> Optional[CricketApiClient]:
    try:
        return CricketApiClient()
    except CricketApiAuthError as e:
        logger.warning(f"CricketApiClient auth error: {e}")
        return None


# ==========================================
# 1. Health Endpoint
# ==========================================
@app.route("/api/health", methods=["GET"])
def health():
    db_info = check_db_connection()
    return jsonify(
        {
            "status": "ok",
            "project": "T20 Cricket Win Predictor",
            "model_loaded": model is not None,
            "database_connected": db_info.get("connected", False),
            "database_type": db_info.get("type", "unknown"),
            "version": "1.0.0",
        }
    )


# ==========================================
# 2. Live Matches Endpoint (T20 Only)
# ==========================================
@app.route("/api/matches", methods=["GET"])
def get_live_matches():
    """
    Returns available live T20/T20I matches from the real Cricket Data API.
    Non-T20 matches (Test, ODI) are strictly filtered out.
    """
    include_recent = request.args.get("recent", "false").lower() == "true"
    client = get_api_client()

    if client is None:
        return jsonify(
            {
                "matches": [],
                "count": 0,
                "message": "CRICKET_API_KEY is not configured on backend.",
                "is_live": False,
            }
        ), 503

    try:
        if include_recent:
            raw_matches = client.get_matches()
        else:
            raw_matches = client.get_current_matches()

        all_norm = [normalize_match_dict(m) for m in raw_matches]
        t20_matches = filter_t20_matches(all_norm)

        # LIVE endpoint: exclude matches that are already completed.
        # /currentMatches can return recently-finished matches (matchEnded=True
        # or status text contains a result like "won by").  They must never appear
        # in the Live Matches section.
        if not include_recent:
            display_matches = [m for m in t20_matches if not is_completed_match(m)]
        else:
            display_matches = t20_matches

        match_list = []
        for m in display_matches:
            match_list.append(
                {
                    "match_id": m.match_id,
                    "name": m.name,
                    "format": m.match_type.upper() if m.match_type else "T20",
                    "status": m.status,
                    "normalized_status": normalize_match_status(m),
                    "venue": m.venue,
                    "teams": m.teams,
                    "match_started": m.match_started,
                    "match_ended": m.match_ended,
                    "innings_count": len(m.innings),
                }
            )

        msg = (
            "Live T20 matches retrieved."
            if match_list
            else "No live T20/T20I matches are currently available."
        )

        return jsonify(
            {
                "matches": match_list,
                "count": len(match_list),
                "message": msg,
                "is_live": not include_recent and len(match_list) > 0,
            }
        )

    except CricketApiRateLimitError as e:
        logger.warning(f"Rate limit reached: {e}")
        return jsonify(
            {
                "matches": [],
                "count": 0,
                "message": "API daily hit limit reached.",
                "error": "RATE_LIMIT_EXCEEDED",
            }
        ), 429
    except CricketApiError as e:
        logger.error(f"CricketApiError in /api/matches: {e}")
        return jsonify(
            {
                "matches": [],
                "count": 0,
                "message": "Failed to connect to cricket data provider.",
                "error": str(e),
            }
        ), 502
    except Exception as e:
        logger.error(f"Unexpected error in /api/matches: {e}")
        return jsonify(
            {
                "matches": [],
                "count": 0,
                "message": "An internal server error occurred.",
            }
        ), 500


# ==========================================
# 3. Match State & Win Probability Endpoint
# ==========================================
@app.route("/api/matches/<match_id>", methods=["GET"])
def get_match_detail(match_id: str):
    """
    Fetches real live match state for a T20 match, evaluates the frozen
    Logistic Regression model, and returns live win probability and match state.
    """
    client = get_api_client()
    if client is None:
        return jsonify({"error": "CRICKET_API_KEY is not configured."}), 503

    try:
        raw = client.get_match_info(match_id)
        if not raw:
            return jsonify({"error": f"Match ID '{match_id}' not found."}), 404

        norm_match = normalize_match_dict(raw)

        # Ensure match is T20
        if not is_t20_match(norm_match):
            return jsonify(
                {
                    "available": False,
                    "reason": f"Match format is '{norm_match.match_type}'. Only T20 matches are supported.",
                    "match": {
                        "match_id": norm_match.match_id,
                        "name": norm_match.name,
                        "format": norm_match.match_type.upper(),
                        "status": norm_match.status,
                    },
                }
            )

        features, reason = extract_match_state_features(norm_match)
        if features is None:
            return jsonify(
                {
                    "available": False,
                    "reason": reason,
                    "match": {
                        "match_id": norm_match.match_id,
                        "name": norm_match.name,
                        "format": norm_match.match_type.upper(),
                        "status": norm_match.status,
                    },
                }
            )

        # Run model inference
        prediction = predict_live_state(features, norm_match, model=model)
        tracker = get_or_create_tracker(match_id)
        signature = tracker.get_state_signature(norm_match)
        is_new, prob_change, abs_change = tracker.update(prediction, signature)

        overs_disp = (
            f"{prediction.overs_bowled_str} ov ({prediction.overs_completed:.2f} fractional)"
            if prediction.overs_bowled_str
            else f"{prediction.overs_completed:.1f} ov"
        )

        return jsonify(
            {
                "available": True,
                "is_demo": False,
                "match_id": prediction.match_id,
                "match_name": prediction.match_name,
                "format": norm_match.match_type.upper(),
                "status": norm_match.status,
                "venue": norm_match.venue,
                "chasing_team": prediction.chasing_team,
                "defending_team": prediction.defending_team,
                "current_score": prediction.current_score,
                "wickets_lost": prediction.wickets_lost,
                "overs_completed": prediction.overs_completed,
                "overs_bowled_str": prediction.overs_bowled_str,
                "overs_display": overs_disp,
                "target_score": prediction.target_score,
                "runs_remaining": prediction.runs_remaining,
                "balls_remaining": prediction.balls_remaining,
                "current_run_rate": prediction.current_run_rate,
                "required_run_rate": prediction.required_run_rate,
                "win_probability": prediction.win_probability,
                "loss_probability": prediction.loss_probability,
                "win_probability_pct": prediction.win_probability_pct,
                "loss_probability_pct": round(prediction.loss_probability * 100, 1),
                "is_terminal": prediction.is_terminal,
                "terminal_state": prediction.terminal_state,
                "win_probability_change": prob_change,
                "absolute_probability_swing": abs_change,
                "is_cached": not is_new,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )

    except CricketApiError as e:
        logger.error(f"Error fetching match {match_id}: {e}")
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        logger.error(f"Unexpected error in /api/matches/{match_id}: {e}")
        return jsonify({"error": "Failed to process match state."}), 500


# ==========================================
# 4. What-If Next-Over Simulation Endpoint
# ==========================================
@app.route("/api/simulation", methods=["POST"])
def run_simulation():
    """
    Evaluates what-if next-over scenarios using the Phase 6 simulation engine.
    """
    data = request.get_json() or {}
    try:
        target_score = float(data.get("target_score", 190.0))
        current_score = float(data.get("current_score", 142.0))
        wickets_lost = int(data.get("wickets_lost", 4))
        balls_remaining = int(data.get("balls_remaining", 24))
        overs_completed = float(data.get("overs_completed", 16.0))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input parameters: {e}"}), 400

    cur_state, scenarios_df = simulate_next_over(
        target_score=target_score,
        current_score=current_score,
        wickets_lost=wickets_lost,
        balls_remaining=balls_remaining,
        overs_completed=overs_completed,
        model=model,
    )

    scenarios_list = scenarios_df.to_dict(orient="records")

    return jsonify(
        {
            "current_state": cur_state,
            "scenarios": scenarios_list,
        }
    )



# ==========================================
# 4.5. Persistent Recent Matches Endpoints (Phase 12)
# STRICT INVARIANT: ZERO Cricket Data API calls.
# Reads directly and exclusively from PostgreSQL / SQLite.
# ==========================================
@app.route("/api/recent/matches", methods=["GET"])
def get_recent_matches():
    """
    Returns stored completed T20/T20I matches from persistent database (Neon PostgreSQL or SQLite).
    STRICT REQUIREMENT: Does NOT call Cricket Data API.
    Enforces Phase 12.3: returns at most the two most recent stored matches.
    """
    try:
        for session in get_db_session():
            matches = (
                session.query(Match)
                .order_by(Match.stored_at.desc(), Match.match_date.desc())
                .limit(2)
                .all()
            )
            result = []
            for m in matches:
                d = m.to_dict()
                d["is_recent"] = True
                d["is_live"] = False
                d["deliveries_count"] = session.query(Delivery).filter_by(match_id=m.match_id).count()
                d["states_count"] = session.query(MatchState).filter_by(match_id=m.match_id).count()
                result.append(d)

            msg = (
                f"Retrieved {len(result)} stored completed T20 matches."
                if result
                else "No stored recent matches found in database. Run sync first."
            )
            return jsonify(
                {
                    "matches": result,
                    "count": len(result),
                    "message": msg,
                    "mode": "RECENT MATCHES",
                    "is_live": False,
                }
            )
    except Exception as e:
        logger.error(f"Error querying recent matches from database: {e}")
        return jsonify({"error": "Failed to query recent matches from database.", "matches": [], "count": 0}), 500


@app.route("/api/recent/matches/<match_id>", methods=["GET"])
def get_recent_match_detail(match_id: str):
    """
    Returns historical ball-by-ball analysis, win probability timeline, swings,
    and simulation scenarios for a stored T20 match.
    STRICT REQUIREMENT: Does NOT call Cricket Data API.
    """
    try:
        for session in get_db_session():
            m = session.query(Match).filter_by(match_id=match_id).first()
            if not m:
                return jsonify({"error": f"Recent match '{match_id}' not found in database."}), 404

            m_dict = m.to_dict()

            # If analysis is not available (e.g. provider had no ball-by-ball data)
            if not m.analysis_available:
                return jsonify(
                    {
                        "available": False,
                        "analysis_available": False,
                        "is_demo": False,
                        "is_recent": True,
                        "mode": "RECENT MATCHES",
                        "match_id": m.match_id,
                        "match_name": m.name,
                        "format": m.format,
                        "status": m.status,
                        "venue": m.venue,
                        "reason": "Historical probability replay unavailable for this match because ball-by-ball data was not provided by the data source.",
                        "match": m_dict,
                        "match_metadata": m_dict,
                        "data_availability_status": {
                            "analysis_available": False,
                            "reason": "Historical probability replay unavailable for this match because ball-by-ball data was not provided by the data source.",
                            "deliveries_count": 0,
                            "states_count": 0,
                        },
                    }
                )

            # Fetch deliveries and second-innings states
            deliveries = (
                session.query(Delivery)
                .filter_by(match_id=match_id)
                .order_by(Delivery.innings.asc(), Delivery.over_number.asc(), Delivery.ball_number.asc())
                .all()
            )

            states = (
                session.query(MatchState)
                .filter_by(match_id=match_id, innings=2)
                .order_by(MatchState.legal_balls_completed.asc())
                .all()
            )

            if not states:
                return jsonify(
                    {
                        "available": False,
                        "analysis_available": False,
                        "is_demo": False,
                        "is_recent": True,
                        "mode": "RECENT MATCHES",
                        "match_id": m.match_id,
                        "match_name": m.name,
                        "format": m.format,
                        "status": m.status,
                        "venue": m.venue,
                        "reason": "No second-innings state progression found.",
                        "match": m_dict,
                        "match_metadata": m_dict,
                    }
                )

            # Latest state
            latest_state = states[-1]

            # Build timeline: sample at over boundaries or every over + latest
            timeline = []
            overs_seen = set()
            for st in states:
                is_over_end = (st.legal_balls_completed % 6 == 0 and st.legal_balls_completed > 0)
                is_last = (st == latest_state)
                over_val = round(st.overs_completed, 1)

                if (is_over_end and over_val not in overs_seen) or is_last:
                    overs_seen.add(over_val)
                    event_str = f"{over_val} ov: {int(st.current_score)}/{st.wickets_lost}"
                    if st.balls_remaining > 0 and st.runs_remaining > 0:
                        event_str += f" (Need {int(st.runs_remaining)} off {st.balls_remaining})"
                    elif st.runs_remaining <= 0:
                        event_str += " (Target reached)"
                    elif st.wickets_lost >= 10:
                        event_str += " (All out)"
                    timeline.append(
                        {
                            "over": over_val,
                            "win_prob": round(st.win_probability * 100, 1),
                            "event": event_str,
                        }
                    )

            deliv_by_id = {d.id: d for d in deliveries if d.innings == 2}

            # Helper for swing event text
            def _build_event_label(st, d_obj):
                if d_obj:
                    if d_obj.wickets > 0:
                        bat = d_obj.batter or "Batter"
                        bwl = d_obj.bowler or "Bowler"
                        return f"WICKET: {bat} b {bwl}"
                    elif d_obj.runs_total >= 6:
                        bat = d_obj.batter or "Batter"
                        bwl = d_obj.bowler or "Bowler"
                        return f"MAXIMUM SIX: {bat} off {bwl}"
                    elif d_obj.runs_total >= 4:
                        bat = d_obj.batter or "Batter"
                        bwl = d_obj.bowler or "Bowler"
                        return f"BOUNDARY FOUR: {bat} off {bwl}"
                    elif d_obj.runs_total == 0:
                        bwl = d_obj.bowler or "Bowler"
                        return f"DOT BALL by {bwl}"
                    else:
                        r = d_obj.runs_total
                        return f"{r} RUN{'S' if r > 1 else ''} ({int(st.current_score)}/{st.wickets_lost})"
                return f"{int(st.current_score)}/{st.wickets_lost} (Need {int(st.runs_remaining)})"

            # Build swings list
            swings_candidates = [
                s for s in states
                if s.probability_swing is not None and abs(s.probability_swing) >= 0.005
            ]

            chasing_team = m.team_2 or "Team 2"
            defending_team = m.team_1 or "Team 1"

            # Compute innings 1 actual wickets from deliveries if available
            inn1_deliveries = [d for d in deliveries if d.innings == 1]
            defending_wickets = sum(d.wickets for d in inn1_deliveries) if inn1_deliveries else 0

            recent_swings = []
            for s in swings_candidates[-5:]:
                s_deliv = deliv_by_id.get(s.delivery_id)
                swing_val_pp = round(s.probability_swing * 100, 1)
                swing_str = f"+{swing_val_pp} pp" if swing_val_pp > 0 else f"{swing_val_pp} pp"
                swing_type = "positive" if swing_val_pp > 0 else "negative" if swing_val_pp < 0 else "neutral"
                benefiting_team = chasing_team if s.probability_swing > 0 else defending_team
                recent_swings.append(
                    {
                        "delivery": (
                            f"{s_deliv.over_number}.{s_deliv.ball_number}"
                            if s_deliv else f"Over {s.overs_completed:.1f}"
                        ),
                        "event": _build_event_label(s, s_deliv),
                        "swing": swing_str,
                        "type": swing_type,
                        "swing_value": s.probability_swing,
                        "benefiting_team": benefiting_team,
                        "state_id": s.id,
                        "delivery_id": s.delivery_id,
                    }
                )

            # Significant swing events: top absolute magnitude swings across ENTIRE match
            sorted_swings = sorted(
                [s for s in states if s.probability_swing is not None],
                key=lambda s: abs(s.probability_swing),
                reverse=True,
            )
            significant_swings = []
            for s in sorted_swings[:6]:
                s_deliv = deliv_by_id.get(s.delivery_id)
                s_val_pp = round(s.probability_swing * 100, 1)
                benefiting_team = chasing_team if s.probability_swing > 0 else defending_team
                significant_swings.append(
                    {
                        "delivery": (
                            f"{s_deliv.over_number}.{s_deliv.ball_number}"
                            if s_deliv else f"Over {s.overs_completed:.1f}"
                        ),
                        "event": _build_event_label(s, s_deliv),
                        "swing": f"+{s_val_pp} pp" if s_val_pp > 0 else f"{s_val_pp} pp",
                        "absolute_swing": round(abs(s.probability_swing) * 100, 1),
                        "type": "positive" if s_val_pp > 0 else "negative",
                        "benefiting_team": benefiting_team,
                        "state_id": s.id,
                        "delivery_id": s.delivery_id,
                    }
                )

            # Run simulation for current state (using frozen Phase 6 simulation engine)
            _, scenarios_df = simulate_next_over(
                target_score=latest_state.target_score,
                current_score=latest_state.current_score,
                wickets_lost=latest_state.wickets_lost,
                balls_remaining=latest_state.balls_remaining,
                overs_completed=latest_state.overs_completed,
                model=model,
            )

            overs_disp = f"{int(latest_state.legal_balls_completed // 6)}.{latest_state.legal_balls_completed % 6} ov ({latest_state.overs_completed:.2f} fractional)"

            chasing_team = m.team_2 or "Team 2"
            defending_team = m.team_1 or "Team 1"

            final_score_dict = {
                "runs": int(latest_state.current_score),
                "wickets": latest_state.wickets_lost,
                "overs": latest_state.overs_completed,
                "target": int(latest_state.target_score),
            }

            innings_info = [
                {"inning": 1, "target": int(latest_state.target_score)},
                {
                    "inning": 2,
                    "runs": int(latest_state.current_score),
                    "wickets": latest_state.wickets_lost,
                    "overs": latest_state.overs_completed,
                },
            ]

            return jsonify(
                {
                    "available": True,
                    "analysis_available": True,
                    "is_demo": False,
                    "is_recent": True,
                    "mode": "RECENT MATCHES",
                    "match_id": m.match_id,
                    "match_name": m.name,
                    "match": m_dict,
                    "match_metadata": m_dict,
                    "format": m.format,
                    "status": m.status,
                    "venue": m.venue,
                    "chasing_team": chasing_team,
                    "defending_team": defending_team,
                    "defending_wickets": defending_wickets,
                    "final_score": final_score_dict,
                    "innings_information": innings_info,
                    "current_score": latest_state.current_score,
                    "wickets_lost": latest_state.wickets_lost,
                    "overs_completed": latest_state.overs_completed,
                    "overs_bowled_str": f"{int(latest_state.legal_balls_completed // 6)}.{latest_state.legal_balls_completed % 6}",
                    "overs_display": overs_disp,
                    "target_score": latest_state.target_score,
                    "runs_remaining": latest_state.runs_remaining,
                    "balls_remaining": latest_state.balls_remaining,
                    "current_run_rate": latest_state.current_run_rate,
                    "required_run_rate": latest_state.required_run_rate,
                    "win_probability": latest_state.win_probability,
                    "current_probability": latest_state.win_probability,
                    "current_win_probability": latest_state.win_probability,
                    "final_win_probability": latest_state.win_probability,
                    "loss_probability": round(1.0 - latest_state.win_probability, 4),
                    "win_probability_pct": round(latest_state.win_probability * 100, 1),
                    "loss_probability_pct": round((1.0 - latest_state.win_probability) * 100, 1),
                    "is_terminal": latest_state.runs_remaining <= 0 or latest_state.wickets_lost >= 10 or latest_state.balls_remaining <= 0,
                    "terminal_state": (
                        "CHASE_SUCCESS" if latest_state.runs_remaining <= 0
                        else "ALL_OUT" if latest_state.wickets_lost >= 10
                        else "BALLS_EXHAUSTED" if latest_state.balls_remaining <= 0
                        else None
                    ),
                    "win_probability_change": latest_state.probability_swing,
                    "absolute_probability_swing": (
                        abs(latest_state.probability_swing) if latest_state.probability_swing is not None else None
                    ),
                    "timeline": timeline,
                    "win_probability_timeline": timeline,
                    "recent_swings": recent_swings,
                    "win_probability_swings": recent_swings,
                    "significant_swing_events": significant_swings,
                    "ball_by_ball_states": [st.to_dict() for st in states],
                    "scenarios": scenarios_df.to_dict(orient="records"),
                    "model_information": {
                        "name": "Logistic Regression (Phase 3 Frozen Baseline)",
                        "features": FEATURE_COLS,
                        "features_count": len(FEATURE_COLS),
                        "features_order": FEATURE_COLS,
                    },
                    "data_availability_status": {
                        "analysis_available": True,
                        "deliveries_count": len(deliveries),
                        "states_count": len(states),
                    },
                    "last_updated": m.stored_at.isoformat() if m.stored_at else datetime.now(timezone.utc).isoformat(),
                }
            )
    except Exception as e:
        logger.error(f"Error querying match {match_id} from database: {e}")
        return jsonify({"error": f"Failed to retrieve match {match_id} from database."}), 500


# ==========================================
# 5. Demo / Simulation Mode Endpoints
# Clearly labeled DEMO / SIMULATION MODE
# ==========================================
DEMO_MATCHES = [
    {
        "match_id": "demo-match-1",
        "name": "India vs Australia, T20 Super 8",
        "format": "T20",
        "status": "India need 48 runs from 24 balls",
        "venue": "Daren Sammy National Cricket Stadium, St Lucia",
        "teams": ["Australia", "India"],
        "is_demo": True,
        "state": {
            "chasing_team": "India",
            "defending_team": "Australia",
            "current_score": 142.0,
            "wickets_lost": 4,
            "overs_completed": 16.0,
            "overs_bowled_str": "16.0",
            "target_score": 190.0,
            "runs_remaining": 48.0,
            "balls_remaining": 24,
            "current_run_rate": 8.88,
            "required_run_rate": 12.00,
            "win_probability": 0.4060,
            "loss_probability": 0.5940,
            "win_probability_pct": 40.6,
            "loss_probability_pct": 59.4,
            "win_probability_change": -0.042,
            "timeline": [
                {"over": 10.0, "win_prob": 52.0, "event": "10 ov: 85/2"},
                {"over": 11.0, "win_prob": 49.5, "event": "11 ov: 92/2"},
                {"over": 12.0, "win_prob": 44.0, "event": "Wicket (98/3)"},
                {"over": 13.0, "win_prob": 51.2, "event": "13 ov: 112/3 (+14 runs)"},
                {"over": 14.0, "win_prob": 48.0, "event": "14 ov: 120/3"},
                {"over": 15.0, "win_prob": 44.8, "event": "Wicket (128/4)"},
                {"over": 16.0, "win_prob": 40.6, "event": "16 ov: 142/4 (Need 48 off 24)"},
            ],
            "recent_swings": [
                {"delivery": "Over 14.3", "event": "BOUNDARY FOUR", "swing": "+4.8%", "type": "positive"},
                {"delivery": "Over 14.6", "event": "WICKET", "swing": "-7.4%", "type": "negative"},
                {"delivery": "Over 15.2", "event": "MAXIMUM SIX", "swing": "+8.2%", "type": "positive"},
                {"delivery": "Over 15.5", "event": "DOT BALL", "swing": "-1.8%", "type": "negative"},
                {"delivery": "Over 16.0", "event": "SINGLE", "swing": "+0.4%", "type": "neutral"},
            ],
        },
    },
    {
        "match_id": "demo-match-2",
        "name": "England vs South Africa, T20 World Cup",
        "format": "T20",
        "status": "South Africa need 28 runs from 18 balls",
        "venue": "Kensington Oval, Bridgetown, Barbados",
        "teams": ["England", "South Africa"],
        "is_demo": True,
        "state": {
            "chasing_team": "South Africa",
            "defending_team": "England",
            "current_score": 148.0,
            "wickets_lost": 4,
            "overs_completed": 17.0,
            "overs_bowled_str": "17.0",
            "target_score": 176.0,
            "runs_remaining": 28.0,
            "balls_remaining": 18,
            "current_run_rate": 8.71,
            "required_run_rate": 9.33,
            "win_probability": 0.6120,
            "loss_probability": 0.3880,
            "win_probability_pct": 61.2,
            "loss_probability_pct": 38.8,
            "win_probability_change": +0.075,
            "timeline": [
                {"over": 12.0, "win_prob": 42.0, "event": "12 ov: 94/3"},
                {"over": 13.0, "win_prob": 45.5, "event": "13 ov: 104/3"},
                {"over": 14.0, "win_prob": 41.0, "event": "Wicket (110/4)"},
                {"over": 15.0, "win_prob": 49.0, "event": "15 ov: 125/4"},
                {"over": 16.0, "win_prob": 53.7, "event": "16 ov: 136/4"},
                {"over": 17.0, "win_prob": 61.2, "event": "17 ov: 148/4 (Need 28 off 18)"},
            ],
            "recent_swings": [
                {"delivery": "Over 15.3", "event": "FOUR OFF COVER", "swing": "+5.2%", "type": "positive"},
                {"delivery": "Over 16.1", "event": "DOT BALL", "swing": "-2.1%", "type": "negative"},
                {"delivery": "Over 16.4", "event": "SIX OVER LONG-ON", "swing": "+9.6%", "type": "positive"},
                {"delivery": "Over 17.0", "event": "TWO RUNS", "swing": "+1.2%", "type": "positive"},
            ],
        },
    },
]


@app.route("/api/demo/matches", methods=["GET"])
def get_demo_matches():
    """Returns curated demo T20 fixtures for testing and presentation."""
    matches = [
        {
            "match_id": m["match_id"],
            "name": m["name"],
            "format": m["format"],
            "status": m["status"],
            "venue": m["venue"],
            "teams": m["teams"],
            "is_demo": True,
        }
        for m in DEMO_MATCHES
    ]
    return jsonify(
        {
            "matches": matches,
            "count": len(matches),
            "mode": "DEMO / SIMULATION MODE",
            "is_live": False,
        }
    )


@app.route("/api/demo/matches/<demo_id>", methods=["GET"])
def get_demo_match_detail(demo_id: str):
    """Returns detailed state, probabilities, and timeline for a demo fixture."""
    selected = next((m for m in DEMO_MATCHES if m["match_id"] == demo_id), None)
    if not selected:
        return jsonify({"error": f"Demo match '{demo_id}' not found."}), 404

    s = selected["state"]

    # Calculate live simulator scenarios using the real Phase 6 simulation engine!
    _, scenarios_df = simulate_next_over(
        target_score=s["target_score"],
        current_score=s["current_score"],
        wickets_lost=s["wickets_lost"],
        balls_remaining=s["balls_remaining"],
        overs_completed=s["overs_completed"],
        model=model,
    )

    overs_disp = f"{s['overs_bowled_str']} ov ({s['overs_completed']:.2f} fractional)"

    return jsonify(
        {
            "available": True,
            "is_demo": True,
            "mode": "DEMO / SIMULATION MODE",
            "match_id": selected["match_id"],
            "match_name": selected["name"],
            "format": selected["format"],
            "status": selected["status"],
            "venue": selected["venue"],
            "chasing_team": s["chasing_team"],
            "defending_team": s["defending_team"],
            "current_score": s["current_score"],
            "wickets_lost": s["wickets_lost"],
            "overs_completed": s["overs_completed"],
            "overs_bowled_str": s["overs_bowled_str"],
            "overs_display": overs_disp,
            "target_score": s["target_score"],
            "runs_remaining": s["runs_remaining"],
            "balls_remaining": s["balls_remaining"],
            "current_run_rate": s["current_run_rate"],
            "required_run_rate": s["required_run_rate"],
            "win_probability": s["win_probability"],
            "loss_probability": s["loss_probability"],
            "win_probability_pct": s["win_probability_pct"],
            "loss_probability_pct": s["loss_probability_pct"],
            "is_terminal": False,
            "terminal_state": None,
            "win_probability_change": s["win_probability_change"],
            "absolute_probability_swing": abs(s["win_probability_change"]),
            "timeline": s["timeline"],
            "recent_swings": s["recent_swings"],
            "scenarios": scenarios_df.to_dict(orient="records"),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    logger.info(f"Starting T20 Win Predictor Flask Backend on port {port} (debug={debug_mode})...")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
