"""
Local End-to-End Smoke Test for Phase 10 Production Readiness.
Verifies all backend HTTP endpoints, responses, JSON contracts, and error handling.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test")


def run_smoke_test():
    client = app.test_client()
    client.testing = True

    results = {}
    all_passed = True

    # 1. GET /api/health
    logger.info("Testing GET /api/health...")
    resp = client.get("/api/health")
    data = resp.get_json()
    status_ok = resp.status_code == 200 and data.get("status") == "ok" and data.get("model_loaded") is True
    results["health"] = {"status_code": resp.status_code, "passed": status_ok, "data": data}
    if not status_ok: all_passed = False

    # 2. GET /api/matches
    logger.info("Testing GET /api/matches...")
    resp = client.get("/api/matches")
    data = resp.get_json()
    matches_ok = resp.status_code == 200 and "matches" in data and isinstance(data["matches"], list)
    results["matches"] = {"status_code": resp.status_code, "passed": matches_ok, "matches_count": len(data.get("matches", [])), "is_live": data.get("is_live")}
    if not matches_ok: all_passed = False

    # 3. GET /api/demo/matches
    logger.info("Testing GET /api/demo/matches...")
    resp = client.get("/api/demo/matches")
    data = resp.get_json()
    demo_ok = resp.status_code == 200 and data.get("mode") == "DEMO / SIMULATION MODE" and len(data.get("matches", [])) > 0
    results["demo_matches"] = {"status_code": resp.status_code, "passed": demo_ok, "matches_count": len(data.get("matches", []))}
    if not demo_ok: all_passed = False

    # 4. GET /api/demo/matches/demo-match-1
    logger.info("Testing GET /api/demo/matches/demo-match-1...")
    resp = client.get("/api/demo/matches/demo-match-1")
    data = resp.get_json()
    demo_detail_ok = (
        resp.status_code == 200
        and data.get("is_demo") is True
        and data.get("chasing_team") == "India"
        and "win_probability" in data
        and "timeline" in data
        and "recent_swings" in data
        and "scenarios" in data
        and "last_updated" in data
    )
    results["demo_match_detail"] = {
        "status_code": resp.status_code,
        "passed": demo_detail_ok,
        "win_probability": data.get("win_probability"),
        "scenarios_count": len(data.get("scenarios", [])),
    }
    if not demo_detail_ok: all_passed = False

    # 5. POST /api/simulation
    logger.info("Testing POST /api/simulation...")
    payload = {
        "target_score": 190.0,
        "current_score": 142.0,
        "wickets_lost": 4,
        "balls_remaining": 24,
        "overs_completed": 16.0,
    }
    resp = client.post("/api/simulation", data=json.dumps(payload), content_type="application/json")
    data = resp.get_json()
    sim_ok = (
        resp.status_code == 200
        and "current_state" in data
        and "scenarios" in data
        and len(data["scenarios"]) == 11
    )
    results["simulation"] = {
        "status_code": resp.status_code,
        "passed": sim_ok,
        "scenarios_count": len(data.get("scenarios", [])),
    }
    if not sim_ok: all_passed = False

    # 6. GET /api/recent/matches (Phase 12 Persistent Storage)
    logger.info("Testing GET /api/recent/matches...")
    resp = client.get("/api/recent/matches")
    data = resp.get_json()
    recent_ok = resp.status_code == 200 and data.get("mode") == "RECENT MATCHES" and "matches" in data
    results["recent_matches"] = {"status_code": resp.status_code, "passed": recent_ok, "matches_count": len(data.get("matches", []))}
    if not recent_ok: all_passed = False

    # 7. CORS Options Preflight Check
    logger.info("Testing CORS preflight on /api/matches...")
    resp = client.options("/api/matches")
    cors_ok = resp.status_code == 200 and "Access-Control-Allow-Origin" in resp.headers
    results["cors"] = {"status_code": resp.status_code, "passed": cors_ok}
    if not cors_ok: all_passed = False

    logger.info(f"Smoke test complete. Overall passed: {all_passed}")
    return {"all_passed": all_passed, "endpoints": results}


if __name__ == "__main__":
    out = run_smoke_test()
    print("\n--- SMOKE TEST SUMMARY ---")
    print(json.dumps(out, indent=2))
    if not out["all_passed"]:
        sys.exit(1)
