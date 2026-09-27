"""
Unit tests for the Flask Backend API (Phase 8).
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from backend.app import app


class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_health_endpoint(self):
        """Test GET /api/health returns 200 and model_loaded: true."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("project"), "T20 Cricket Win Predictor")
        self.assertTrue(data.get("model_loaded"))

    @patch("backend.app.get_api_client")
    def test_matches_t20_filtering(self, mock_client_factory):
        """Test GET /api/matches returns ONLY T20/T20I matches and rejects Test/ODI."""
        mock_client = MagicMock()
        mock_client.get_current_matches.return_value = [
            {"id": "m1", "name": "Test Match", "matchType": "test", "status": "Day 1"},
            {"id": "m2", "name": "ODI Match", "matchType": "odi", "status": "Innings break"},
            {"id": "m3", "name": "T20 Match", "matchType": "t20", "status": "Live"},
            {"id": "m4", "name": "T20I Match", "matchType": "t20i", "status": "Live"},
        ]
        mock_client_factory.return_value = mock_client

        resp = self.client.get("/api/matches")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        matches = data.get("matches", [])
        self.assertEqual(len(matches), 2)
        match_ids = [m["match_id"] for m in matches]
        self.assertIn("m3", match_ids)
        self.assertIn("m4", match_ids)
        self.assertNotIn("m1", match_ids)
        self.assertNotIn("m2", match_ids)

    @patch("backend.app.get_api_client")
    def test_matches_empty_when_no_live_t20(self, mock_client_factory):
        """Test GET /api/matches handles no live T20 matches without crashing."""
        mock_client = MagicMock()
        # All ongoing matches are tests
        mock_client.get_current_matches.return_value = [
            {"id": "m1", "name": "County Championship Test", "matchType": "test", "status": "Day 2: Stumps"}
        ]
        mock_client_factory.return_value = mock_client

        resp = self.client.get("/api/matches")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("matches"), [])
        self.assertEqual(data.get("count"), 0)
        self.assertIn("No live T20/T20I matches are currently available.", data.get("message"))
        self.assertFalse(data.get("is_live"))

    def test_simulation_endpoint(self):
        """Test POST /api/simulation returns what-if next-over scenarios."""
        payload = {
            "target_score": 190.0,
            "current_score": 142.0,
            "wickets_lost": 4,
            "balls_remaining": 24,
            "overs_completed": 16.0,
        }
        resp = self.client.post(
            "/api/simulation",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("current_state", data)
        self.assertIn("scenarios", data)
        scenarios = data.get("scenarios")
        self.assertEqual(len(scenarios), 11)
        # Verify Dot Over probability drops
        dot_over = next(s for s in scenarios if s["scenario_name"] == "Dot Over")
        self.assertLess(dot_over["simulated_win_probability"], data["current_state"]["current_win_probability"])

    def test_demo_matches_endpoints(self):
        """Test demo fixtures endpoints."""
        resp = self.client.get("/api/demo/matches")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertGreater(data.get("count"), 0)
        self.assertEqual(data.get("mode"), "DEMO / SIMULATION MODE")

        # Detail endpoint
        detail_resp = self.client.get("/api/demo/matches/demo-match-1")
        self.assertEqual(detail_resp.status_code, 200)
        detail_data = detail_resp.get_json()
        self.assertTrue(detail_data.get("is_demo"))
        self.assertEqual(detail_data.get("mode"), "DEMO / SIMULATION MODE")
        self.assertEqual(detail_data.get("chasing_team"), "India")
        self.assertIn("timeline", detail_data)
        self.assertIn("recent_swings", detail_data)
        self.assertIn("scenarios", detail_data)
        self.assertGreaterEqual(detail_data.get("win_probability"), 0.0)
        self.assertLessEqual(detail_data.get("win_probability"), 1.0)
        self.assertIn("last_updated", detail_data)

    @patch("backend.app.get_api_client")
    def test_matches_client_not_configured_returns_503(self, mock_client_factory):
        """Test GET /api/matches returns 503 when API client cannot initialize."""
        mock_client_factory.return_value = None
        resp = self.client.get("/api/matches")
        self.assertEqual(resp.status_code, 503)
        data = resp.get_json()
        self.assertIn("CRICKET_API_KEY", data.get("message"))

    @patch("backend.app.get_api_client")
    def test_matches_rate_limit_returns_429(self, mock_client_factory):
        """Test GET /api/matches returns 429 when API rate limit is exceeded."""
        from src.live.cricket_api import CricketApiRateLimitError
        mock_client = MagicMock()
        mock_client.get_current_matches.side_effect = CricketApiRateLimitError("Daily quota exhausted")
        mock_client_factory.return_value = mock_client

        resp = self.client.get("/api/matches")
        self.assertEqual(resp.status_code, 429)
        data = resp.get_json()
        self.assertEqual(data.get("error"), "RATE_LIMIT_EXCEEDED")

    @patch("backend.app.get_api_client")
    def test_matches_provider_error_returns_502(self, mock_client_factory):
        """Test GET /api/matches returns 502 when provider returns an error."""
        from src.live.cricket_api import CricketApiError
        mock_client = MagicMock()
        mock_client.get_current_matches.side_effect = CricketApiError("Gateway timeout")
        mock_client_factory.return_value = mock_client

        resp = self.client.get("/api/matches")
        self.assertEqual(resp.status_code, 502)
        data = resp.get_json()
        self.assertIn("Failed to connect to cricket data provider", data.get("message"))

    @patch("backend.app.get_api_client")
    def test_match_detail_not_found_returns_404(self, mock_client_factory):
        """Test GET /api/matches/<id> returns 404 when match does not exist."""
        mock_client = MagicMock()
        mock_client.get_match_info.return_value = None
        mock_client_factory.return_value = mock_client

        resp = self.client.get("/api/matches/non-existent-id")
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn("not found", data.get("error"))

    @patch("backend.app.get_api_client")
    def test_match_detail_non_t20_returns_unavailable(self, mock_client_factory):
        """Test GET /api/matches/<id> safely returns available: False for non-T20."""
        mock_client = MagicMock()
        mock_client.get_match_info.return_value = {
            "id": "test-match-1",
            "name": "England vs Australia Test",
            "matchType": "test",
            "status": "Day 3",
            "teams": ["England", "Australia"],
        }
        mock_client_factory.return_value = mock_client

        resp = self.client.get("/api/matches/test-match-1")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data.get("available"))
        self.assertIn("Only T20 matches are supported", data.get("reason"))

    def test_simulation_invalid_params_returns_400(self):
        """Test POST /api/simulation returns 400 on malformed input data."""
        resp = self.client.post(
            "/api/simulation",
            data=json.dumps({"target_score": "not-a-number"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("Invalid input parameters", data.get("error"))

    def test_cors_options_preflight(self):
        """Test OPTIONS preflight returns proper CORS headers."""
        resp = self.client.options("/api/matches")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Access-Control-Allow-Origin", resp.headers)


if __name__ == "__main__":
    unittest.main()
