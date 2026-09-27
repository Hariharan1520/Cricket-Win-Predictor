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

# Load .env
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()

from src.live.cricket_api import (
    CricketApiClient,
    CricketApiError,
    CricketApiAuthError,
    CricketApiRateLimitError,
)
from src.live.live_features import (
    LiveMatchFeatures,
    NormalizedMatch,
    extract_match_state_features,
    filter_t20_matches,
    is_t20_match,
    normalize_match_dict,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard_backend")

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
    return jsonify(
        {
            "status": "ok",
            "project": "T20 Cricket Win Predictor",
            "model_loaded": model is not None,
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

        match_list = []
        for m in t20_matches:
            match_list.append(
                {
                    "match_id": m.match_id,
                    "name": m.name,
                    "format": m.match_type.upper() if m.match_type else "T20",
                    "status": m.status,
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
