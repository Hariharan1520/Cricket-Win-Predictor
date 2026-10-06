"""
Unit and Integration Test Suite for Live Cricket Integration (Phase 7).

All tests use mocked API data or deterministic offline states.
No real live API quota is consumed during unit tests.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.live.cricket_api import (
    CricketApiClient,
    CricketApiError,
    CricketApiAuthError,
    CricketApiRateLimitError,
    mask_sensitive_url,
)
from src.live.live_features import (
    LiveMatchFeatures,
    NormalizedMatch,
    NormalizedInnings,
    extract_match_state_features,
    filter_t20_matches,
    is_completed_match,
    is_t20_format,
    is_t20_match,
    normalize_match_dict,
    normalize_match_status,
    parse_cricket_overs,
)
from src.live.live_match import (
    LiveStateTracker,
    LivePredictionResult,
    display_match_list,
    load_prediction_model,
    predict_live_state,
)


class TestCricketApiClient(unittest.TestCase):
    """Tests for the Cricket Data API Client."""

    def test_api_key_missing(self):
        """Verify client raises CricketApiAuthError when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(CricketApiAuthError):
                CricketApiClient(api_key=None)

    def test_api_mask_sensitive_url(self):
        """Verify API key is completely redacted in URLs."""
        raw_url = "https://api.cricapi.com/v1/currentMatches?apikey=5fa50043-secret-key-123&offset=0"
        masked = mask_sensitive_url(raw_url)
        self.assertNotIn("5fa50043-secret-key-123", masked)
        self.assertIn("apikey=***REDACTED***", masked)

    @patch("src.live.cricket_api.requests.Session.get")
    def test_api_unavailable_timeout(self, mock_get):
        """Verify timeout is caught and wrapped without exposing API key."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out to api.cricapi.com")
        client = CricketApiClient(api_key="test-key-12345")
        with self.assertRaises(CricketApiError) as ctx:
            client.get_current_matches()
        self.assertIn("timed out", str(ctx.exception).lower())
        self.assertNotIn("test-key-12345", str(ctx.exception))

    @patch("src.live.cricket_api.requests.Session.get")
    def test_malformed_api_response(self, mock_get):
        """Verify invalid/malformed JSON raises CricketApiError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>502 Bad Gateway</body></html>"
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_resp

        client = CricketApiClient(api_key="test-key-12345")
        with self.assertRaises(CricketApiError) as ctx:
            client.get_current_matches()
        self.assertIn("malformed json", str(ctx.exception).lower())

    @patch("src.live.cricket_api.requests.Session.get")
    def test_current_matches_successfully_returned(self, mock_get):
        """Verify successful current matches retrieval."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "data": [
                {
                    "id": "match-uuid-1",
                    "name": "India vs Pakistan, T20",
                    "matchType": "t20",
                    "status": "India won by 5 wickets",
                    "teams": ["India", "Pakistan"],
                }
            ],
        }
        mock_get.return_value = mock_resp

        client = CricketApiClient(api_key="test-key-12345")
        matches = client.get_current_matches()
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["id"], "match-uuid-1")
        self.assertEqual(matches[0]["name"], "India vs Pakistan, T20")


class TestFeatureEngineeringAndParsing(unittest.TestCase):
    """Tests for cricket overs parsing, normalization, and feature extraction."""

    def test_parse_cricket_overs(self):
        """Verify cricket overs notation parsing."""
        # Standard: 18.4 overs = 18 * 6 + 4 = 112 legal balls, 18.6667 fraction
        comp_o, b_in_o, total_b, o_frac = parse_cricket_overs("18.4")
        self.assertEqual(comp_o, 18)
        self.assertEqual(b_in_o, 4)
        self.assertEqual(total_b, 112)
        self.assertAlmostEqual(o_frac, 18.6667, places=3)

        # Maiden completed over: 20.0
        comp_o, b_in_o, total_b, o_frac = parse_cricket_overs(20)
        self.assertEqual(comp_o, 20)
        self.assertEqual(b_in_o, 0)
        self.assertEqual(total_b, 120)
        self.assertEqual(o_frac, 20.0)

        # Zero balls completed: 0.0
        comp_o, b_in_o, total_b, o_frac = parse_cricket_overs(0.0)
        self.assertEqual(total_b, 0)
        self.assertEqual(o_frac, 0.0)

    def test_first_innings_only(self):
        """Verify first-innings state returns unavailable for chase win prediction."""
        raw = {
            "id": "m1",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Team A 120/4 (15.0 ov)",
            "teams": ["Team A", "Team B"],
            "score": [{"r": 120, "w": 4, "o": 15, "inning": "Team A Inning 1"}],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(features)
        self.assertEqual(reason, "Prediction unavailable: second innings has not started.")

    def test_second_innings_in_progress(self):
        """Verify mid-innings 2nd innings features calculation."""
        raw = {
            "id": "m2",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Team B need 48 runs from 24 balls",
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 189, "w": 6, "o": 20, "inning": "Team A Inning 1"},
                {"r": 142, "w": 4, "o": 16, "inning": "Team B Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(reason)
        self.assertIsNotNone(features)
        self.assertEqual(features.target_score, 190.0)
        self.assertEqual(features.current_score, 142.0)
        self.assertEqual(features.wickets_lost, 4)
        self.assertEqual(features.balls_remaining, 24)  # 120 - 96
        self.assertEqual(features.runs_remaining, 48.0)
        self.assertEqual(features.overs_completed, 16.0)
        self.assertAlmostEqual(features.current_run_rate, 8.875, places=2)
        self.assertAlmostEqual(features.required_run_rate, 12.0, places=2)
        self.assertFalse(features.is_terminal)

        # Check DataFrame format
        df = features.to_dataframe()
        self.assertEqual(df.shape, (1, 8))
        self.assertEqual(list(df.columns), [
            "target_score", "current_score", "wickets_lost", "runs_remaining",
            "balls_remaining", "overs_completed", "current_run_rate", "required_run_rate"
        ])

    def test_target_reached_terminal(self):
        """Verify target reached results in terminal 1.0 probability."""
        raw = {
            "id": "m3",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Team B won by 6 wickets",
            "matchEnded": True,
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 150, "w": 8, "o": 20, "inning": "Team A Inning 1"},
                {"r": 152, "w": 4, "o": 18.3, "inning": "Team B Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(reason)
        self.assertTrue(features.is_terminal)
        self.assertEqual(features.terminal_probability, 1.0)
        self.assertEqual(features.terminal_state, "Target Reached (1.0)")

    def test_all_out_terminal(self):
        """Verify 10 wickets lost results in terminal 0.0 probability."""
        raw = {
            "id": "m4",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Team A won by 30 runs",
            "matchEnded": True,
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 160, "w": 6, "o": 20, "inning": "Team A Inning 1"},
                {"r": 130, "w": 10, "o": 17.2, "inning": "Team B Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(reason)
        self.assertTrue(features.is_terminal)
        self.assertEqual(features.terminal_probability, 0.0)
        self.assertEqual(features.terminal_state, "All Out (0.0)")

    def test_balls_exhausted_terminal(self):
        """Verify balls exhausted before target results in terminal 0.0 probability."""
        raw = {
            "id": "m5",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Team A won by 15 runs",
            "matchEnded": True,
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 160, "w": 6, "o": 20, "inning": "Team A Inning 1"},
                {"r": 145, "w": 7, "o": 20, "inning": "Team B Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(reason)
        self.assertTrue(features.is_terminal)
        self.assertEqual(features.terminal_probability, 0.0)
        self.assertEqual(features.terminal_state, "Balls Exhausted (0.0)")

    def test_abandoned_no_result(self):
        """Verify abandoned match returns no valid result message."""
        raw = {
            "id": "m6",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Match abandoned without a ball bowled",
            "teams": ["Team A", "Team B"],
            "score": [],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(features)
        self.assertEqual(reason, "Prediction unavailable: match has no valid result.")

    def test_super_over_unsupported(self):
        """Verify Super Over is safely detected and marked unsupported."""
        raw = {
            "id": "m7",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Match tied (Team A won Super Over)",
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 150, "w": 6, "o": 20, "inning": "Team A Inning 1"},
                {"r": 150, "w": 8, "o": 20, "inning": "Team B Inning 1"},
                {"r": 15, "w": 1, "o": 1, "inning": "Team A Super Over"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(features)
        self.assertIn("Super Over match states are not supported", reason)

    def test_non_t20_format(self):
        """Verify Test and ODI formats are explicitly rejected."""
        raw = {
            "id": "m8",
            "name": "Team A vs Team B, 1st Test",
            "matchType": "test",
            "status": "Day 3: Stumps",
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 350, "w": 10, "o": 110, "inning": "Team A Inning 1"},
                {"r": 200, "w": 10, "o": 70, "inning": "Team B Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(features)
        self.assertIn("match format is 'test', only T20 matches are supported", reason)

    def test_zero_balls_completed_at_innings_start(self):
        """Verify chase start (0.0 overs) calculates correct finite features."""
        raw = {
            "id": "m9",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Innings break / Chase starting",
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 160, "w": 5, "o": 20, "inning": "Team A Inning 1"},
                {"r": 0, "w": 0, "o": 0, "inning": "Team B Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(reason)
        self.assertIsNotNone(features)
        self.assertEqual(features.target_score, 161.0)
        self.assertEqual(features.current_score, 0.0)
        self.assertEqual(features.balls_remaining, 120)
        self.assertEqual(features.current_run_rate, 0.0)
        self.assertAlmostEqual(features.required_run_rate, 161.0 / 20.0, places=2)

    def test_final_over_chase(self):
        """Verify final over (19.2 overs, 4 balls left) feature calculation."""
        raw = {
            "id": "m10",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Need 8 runs from 4 balls",
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 170, "w": 5, "o": 20, "inning": "Team A Inning 1"},
                {"r": 163, "w": 6, "o": 19.2, "inning": "Team B Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(reason)
        self.assertIsNotNone(features)
        self.assertEqual(features.target_score, 171.0)
        self.assertEqual(features.balls_remaining, 4)
        self.assertEqual(features.runs_remaining, 8.0)
        self.assertAlmostEqual(features.required_run_rate, 8.0 / (4 / 6.0), places=2)


class TestModelPredictionAndStateTracking(unittest.TestCase):
    """Tests for model inference and probability-swing tracking."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_prediction_model()

    def test_prediction_output_bounds(self):
        """Verify win probability is always bounded strictly between 0.0 and 1.0."""
        features = LiveMatchFeatures(
            target_score=190.0,
            current_score=142.0,
            wickets_lost=4,
            runs_remaining=48.0,
            balls_remaining=24,
            overs_completed=16.0,
            current_run_rate=8.88,
            required_run_rate=12.00,
            chasing_team="India",
            defending_team="Australia",
        )
        norm_match = NormalizedMatch(
            match_id="test-id",
            name="India vs Australia",
            match_type="t20",
            status="In progress",
            venue="Melbourne",
            date="2026-09-26",
            teams=["India", "Australia"],
            match_started=True,
            match_ended=False,
        )
        pred = predict_live_state(features, norm_match, model=self.model)
        self.assertGreaterEqual(pred.win_probability, 0.0)
        self.assertLessEqual(pred.win_probability, 1.0)
        self.assertAlmostEqual(pred.win_probability + pred.loss_probability, 1.0, places=3)
        self.assertEqual(pred.win_probability_pct, round(pred.win_probability * 100, 1))

    def test_win_probability_change_tracking(self):
        """Verify that state tracker properly computes win probability swings."""
        tracker = LiveStateTracker()

        pred1 = LivePredictionResult(
            match_id="m1",
            match_name="Team A vs Team B",
            chasing_team="Team B",
            defending_team="Team A",
            current_score=100.0,
            wickets_lost=3,
            overs_completed=12.0,
            target_score=160.0,
            runs_remaining=60.0,
            balls_remaining=48,
            current_run_rate=8.33,
            required_run_rate=7.50,
            win_probability=0.5500,
            loss_probability=0.4500,
            win_probability_pct=55.0,
            is_terminal=False,
        )
        sig1 = ("m1", 2, "100/3", False, ((("Team B", 100, 3, 72),)))
        is_new1, chg1, abs1 = tracker.update(pred1, sig1)
        self.assertTrue(is_new1)
        self.assertIsNone(chg1)  # First observation has no delta

        # Delivery 2: Six hit! Probability rises to 0.6500
        pred2 = LivePredictionResult(
            match_id="m1",
            match_name="Team A vs Team B",
            chasing_team="Team B",
            defending_team="Team A",
            current_score=106.0,
            wickets_lost=3,
            overs_completed=12.1667,
            target_score=160.0,
            runs_remaining=54.0,
            balls_remaining=47,
            current_run_rate=8.71,
            required_run_rate=6.89,
            win_probability=0.6500,
            loss_probability=0.3500,
            win_probability_pct=65.0,
            is_terminal=False,
        )
        sig2 = ("m1", 2, "106/3", False, ((("Team B", 106, 3, 73),)))
        is_new2, chg2, abs2 = tracker.update(pred2, sig2)
        self.assertTrue(is_new2)
        self.assertAlmostEqual(chg2, 0.1000, places=4)
        self.assertAlmostEqual(abs2, 0.1000, places=4)

    def test_duplicate_live_state_ignored(self):
        """Verify that identical match state is detected and duplicate update skipped."""
        tracker = LiveStateTracker()

        pred = LivePredictionResult(
            match_id="m1",
            match_name="Team A vs Team B",
            chasing_team="Team B",
            defending_team="Team A",
            current_score=100.0,
            wickets_lost=3,
            overs_completed=12.0,
            target_score=160.0,
            runs_remaining=60.0,
            balls_remaining=48,
            current_run_rate=8.33,
            required_run_rate=7.50,
            win_probability=0.5500,
            loss_probability=0.4500,
            win_probability_pct=55.0,
            is_terminal=False,
        )
        sig = ("m1", 2, "100/3", False, ((("Team B", 100, 3, 72),)))

        # First call: valid new state
        is_new1, _, _ = tracker.update(pred, sig)
        self.assertTrue(is_new1)

        # Second call with identical signature: duplicate state
        is_new2, chg2, abs2 = tracker.update(pred, sig)
        self.assertFalse(is_new2)
        self.assertEqual(chg2, 0.0)
        self.assertEqual(abs2, 0.0)


class TestT20FormatFiltering(unittest.TestCase):
    """Tests for Phase 7.2 strict T20-only filtering."""

    def test_is_t20_format(self):
        """Verify is_t20_format accepts only t20/t20i case-insensitively."""
        # Valid T20 formats
        self.assertTrue(is_t20_format("t20"))
        self.assertTrue(is_t20_format("T20"))
        self.assertTrue(is_t20_format("t20i"))
        self.assertTrue(is_t20_format("T20I"))
        self.assertTrue(is_t20_format(" T20 "))

        # Invalid non-T20 formats
        self.assertFalse(is_t20_format("test"))
        self.assertFalse(is_t20_format("Test"))
        self.assertFalse(is_t20_format("odi"))
        self.assertFalse(is_t20_format("ODI"))
        self.assertFalse(is_t20_format("First-class"))
        self.assertFalse(is_t20_format("List A"))
        self.assertFalse(is_t20_format(""))
        self.assertFalse(is_t20_format(None))

    def test_is_t20_match_variants(self):
        """Verify is_t20_match correctly inspects string, dict, or NormalizedMatch."""
        # String
        self.assertTrue(is_t20_match("t20"))
        self.assertFalse(is_t20_match("test"))

        # Raw Dict from API
        self.assertTrue(is_t20_match({"matchType": "t20"}))
        self.assertTrue(is_t20_match({"matchType": "T20I"}))
        self.assertFalse(is_t20_match({"matchType": "test"}))
        self.assertFalse(is_t20_match({"matchType": "odi"}))
        self.assertFalse(is_t20_match({}))

        # NormalizedMatch object
        norm_t20 = NormalizedMatch(
            match_id="t20-id",
            name="India vs Pakistan",
            match_type="t20",
            status="In progress",
            venue="Melbourne",
            date="2026-09-26",
            teams=["India", "Pakistan"],
            match_started=True,
            match_ended=False,
        )
        self.assertTrue(is_t20_match(norm_t20))

        norm_test = NormalizedMatch(
            match_id="test-id",
            name="England vs Australia",
            match_type="test",
            status="Day 2",
            venue="Lord's",
            date="2026-09-26",
            teams=["England", "Australia"],
            match_started=True,
            match_ended=False,
        )
        self.assertFalse(is_t20_match(norm_test))
        self.assertFalse(is_t20_match(None))

    def test_mixed_api_response_filtering(self):
        """Verify that a mixed API response containing Test, ODI, T20, and T20I is strictly filtered."""
        mixed_matches = [
            NormalizedMatch("m1", "Lancashire vs Durham", "test", "Day 2", "Old Trafford", "2026-09-26", ["Lancashire", "Durham"], True, False),
            NormalizedMatch("m2", "India vs South Africa", "odi", "Innings break", "Johannesburg", "2026-09-26", ["India", "South Africa"], True, False),
            NormalizedMatch("m3", "Mumbai Indians vs CSK", "t20", "Chase in progress", "Wankhede", "2026-09-26", ["MI", "CSK"], True, False),
            NormalizedMatch("m4", "Pakistan vs New Zealand", "t20i", "Need 30 off 18", "Lahore", "2026-09-26", ["Pakistan", "New Zealand"], True, False),
        ]
        filtered = filter_t20_matches(mixed_matches)
        self.assertEqual(len(filtered), 2)
        self.assertEqual([m.match_id for m in filtered], ["m3", "m4"])
        self.assertEqual([m.match_type for m in filtered], ["t20", "t20i"])

    def test_no_live_t20_handling(self):
        """Verify that when only non-T20 matches are present, the application cleanly reports zero live T20s."""
        non_t20_matches = [
            NormalizedMatch("m1", "Match 1", "test", "Day 1", "Venue", "2026-09-26", ["Team A", "Team B"], True, False),
            NormalizedMatch("m2", "Match 2", "odi", "In progress", "Venue", "2026-09-26", ["Team C", "Team D"], True, False),
        ]
        filtered = filter_t20_matches(non_t20_matches)
        self.assertEqual(filtered, [])

        # Verify display output without crashing
        import io
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            display_match_list(filtered, title="T20 LIVE MATCHES")
        output = captured.getvalue()
        self.assertIn("No live T20/T20I matches are currently available.", output)
        self.assertNotIn("Match 1", output)

    def test_display_match_list_formatting(self):
        """Verify display_match_list output format for T20 matches."""
        import io
        t20_matches = [
            NormalizedMatch("m_t20", "India vs Pakistan", "t20", "India need 20 off 12", "Melbourne", "2026-09-26", ["India", "Pakistan"], True, False)
        ]
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            display_match_list(t20_matches, title="T20 LIVE MATCHES")
        output = captured.getvalue()
        self.assertIn("T20 LIVE MATCHES", output)
        self.assertIn("India vs Pakistan", output)
        self.assertIn("Format:   T20", output)
        self.assertIn("Match ID: m_t20", output)


class TestPhase9LiveValidationPipeline(unittest.TestCase):
    """
    Dedicated test suite for Phase 9 Live T20 Match Validation & Data Reliability.
    Verifies all 20 required validation specifications explicitly.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = load_prediction_model()

    def test_01_t20_filtering(self):
        """1. Verify T20 format is accepted case-insensitively and with whitespace."""
        self.assertTrue(is_t20_format("t20"))
        self.assertTrue(is_t20_format("T20"))
        self.assertTrue(is_t20_format("  T20  "))
        self.assertTrue(is_t20_match({"matchType": "t20"}))

    def test_02_t20i_filtering(self):
        """2. Verify T20I format is accepted case-insensitively and with whitespace."""
        self.assertTrue(is_t20_format("t20i"))
        self.assertTrue(is_t20_format("T20I"))
        self.assertTrue(is_t20_format(" t20i "))
        self.assertTrue(is_t20_match({"matchType": "T20I"}))

    def test_03_test_match_rejection(self):
        """3. Verify Test format is strictly rejected."""
        self.assertFalse(is_t20_format("test"))
        self.assertFalse(is_t20_format("Test"))
        self.assertFalse(is_t20_format("TEST"))
        self.assertFalse(is_t20_match({"matchType": "test"}))

    def test_04_odi_rejection(self):
        """4. Verify ODI format is strictly rejected."""
        self.assertFalse(is_t20_format("odi"))
        self.assertFalse(is_t20_format("ODI"))
        self.assertFalse(is_t20_format("One-Day International"))
        self.assertFalse(is_t20_match({"matchType": "odi"}))

    def test_05_super_over_rejection(self):
        """5. Verify Super Over match states are safely detected and rejected."""
        raw = {
            "id": "so-1",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Match tied (Super Over)",
            "teams": ["Team A", "Team B"],
            "score": [
                {"r": 150, "w": 6, "o": 20, "inning": "Team A Inning 1"},
                {"r": 150, "w": 8, "o": 20, "inning": "Team B Inning 1"},
                {"r": 12, "w": 1, "o": 1, "inning": "Team A Super Over"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, reason = extract_match_state_features(norm)
        self.assertIsNone(features)
        self.assertIn("Super Over match states are not supported", reason)

    def test_06_overs_parsing(self):
        """6. Verify cricket overs notation conversion to balls and true fractional overs."""
        # 16.0 = 16 overs = 96 balls
        comp, b_in_o, tot_b, frac = parse_cricket_overs("16.0")
        self.assertEqual(comp, 16)
        self.assertEqual(tot_b, 96)
        self.assertEqual(frac, 16.0)

        # 16.1 = 97 balls
        comp, b_in_o, tot_b, frac = parse_cricket_overs("16.1")
        self.assertEqual(tot_b, 97)
        self.assertAlmostEqual(frac, 16 + 1/6.0, places=4)

        # 16.4 = 100 balls (must NOT be treated as 16.4 decimal!)
        comp, b_in_o, tot_b, frac = parse_cricket_overs("16.4")
        self.assertEqual(tot_b, 100)
        self.assertAlmostEqual(frac, 16.6667, places=3)
        self.assertNotEqual(frac, 16.4)

        # 16.5 = 101 balls
        comp, b_in_o, tot_b, frac = parse_cricket_overs("16.5")
        self.assertEqual(tot_b, 101)
        self.assertAlmostEqual(frac, 16 + 5/6.0, places=4)

        # 17.0 = 102 balls
        comp, b_in_o, tot_b, frac = parse_cricket_overs("17.0")
        self.assertEqual(tot_b, 102)
        self.assertEqual(frac, 17.0)

    def test_07_120_ball_calculation(self):
        """7. Verify strict 120-ball second innings calculation."""
        comp, b_in_o, tot_b, frac = parse_cricket_overs("17.0")
        balls_remaining = 120 - tot_b
        self.assertEqual(balls_remaining, 18)
        self.assertEqual(tot_b + balls_remaining, 120)

    def test_08_target_calculation(self):
        """8. Verify target_score calculation (1st innings score + 1)."""
        raw = {
            "id": "t8",
            "name": "SA vs ENG",
            "matchType": "t20",
            "status": "In progress",
            "teams": ["England", "South Africa"],
            "score": [
                {"r": 175, "w": 6, "o": 20, "inning": "England Inning 1"},
                {"r": 148, "w": 4, "o": 17.0, "inning": "South Africa Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, _ = extract_match_state_features(norm)
        self.assertIsNotNone(features)
        self.assertEqual(features.target_score, 176.0)

    def test_09_runs_remaining(self):
        """9. Verify runs_remaining calculation (target - current)."""
        raw = {
            "id": "t9",
            "name": "SA vs ENG",
            "matchType": "t20",
            "status": "In progress",
            "teams": ["England", "South Africa"],
            "score": [
                {"r": 175, "w": 6, "o": 20, "inning": "England Inning 1"},
                {"r": 148, "w": 4, "o": 17.0, "inning": "South Africa Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, _ = extract_match_state_features(norm)
        self.assertIsNotNone(features)
        self.assertEqual(features.runs_remaining, 28.0)  # 176 - 148

    def test_10_balls_remaining(self):
        """10. Verify balls_remaining calculation from overs bowled."""
        raw = {
            "id": "t10",
            "name": "SA vs ENG",
            "matchType": "t20",
            "status": "In progress",
            "teams": ["England", "South Africa"],
            "score": [
                {"r": 175, "w": 6, "o": 20, "inning": "England Inning 1"},
                {"r": 148, "w": 4, "o": 17.0, "inning": "South Africa Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, _ = extract_match_state_features(norm)
        self.assertIsNotNone(features)
        self.assertEqual(features.balls_remaining, 18)  # 120 - 102

    def test_11_current_run_rate(self):
        """11. Verify current_run_rate calculation (score / overs_completed)."""
        raw = {
            "id": "t11",
            "name": "SA vs ENG",
            "matchType": "t20",
            "status": "In progress",
            "teams": ["England", "South Africa"],
            "score": [
                {"r": 175, "w": 6, "o": 20, "inning": "England Inning 1"},
                {"r": 148, "w": 4, "o": 17.0, "inning": "South Africa Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, _ = extract_match_state_features(norm)
        self.assertIsNotNone(features)
        expected_crr = 148.0 / 17.0
        self.assertAlmostEqual(features.current_run_rate, expected_crr, places=2)

    def test_12_required_run_rate(self):
        """12. Verify required_run_rate calculation (runs_remaining / (balls_remaining / 6))."""
        raw = {
            "id": "t12",
            "name": "SA vs ENG",
            "matchType": "t20",
            "status": "In progress",
            "teams": ["England", "South Africa"],
            "score": [
                {"r": 175, "w": 6, "o": 20, "inning": "England Inning 1"},
                {"r": 148, "w": 4, "o": 17.0, "inning": "South Africa Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, _ = extract_match_state_features(norm)
        self.assertIsNotNone(features)
        expected_rrr = 28.0 / (18 / 6.0)  # 28 / 3.0 = 9.33
        self.assertAlmostEqual(features.required_run_rate, expected_rrr, places=2)

    def test_13_feature_ordering(self):
        """13. Verify exact feature ordering required by scikit-learn model."""
        features = LiveMatchFeatures(
            target_score=176.0, current_score=148.0, wickets_lost=4,
            runs_remaining=28.0, balls_remaining=18, overs_completed=17.0,
            current_run_rate=8.71, required_run_rate=9.33,
            chasing_team="South Africa", defending_team="England",
        )
        df = features.to_dataframe()
        expected_order = [
            "target_score", "current_score", "wickets_lost", "runs_remaining",
            "balls_remaining", "overs_completed", "current_run_rate", "required_run_rate"
        ]
        self.assertEqual(list(df.columns), expected_order)

    def test_14_probability_sum(self):
        """14. Verify chasing_probability + defending_probability equals 1.0."""
        features = LiveMatchFeatures(
            target_score=176.0, current_score=148.0, wickets_lost=4,
            runs_remaining=28.0, balls_remaining=18, overs_completed=17.0,
            current_run_rate=8.71, required_run_rate=9.33,
            chasing_team="South Africa", defending_team="England",
        )
        dummy_match = NormalizedMatch(
            match_id="t14", name="SA vs ENG", match_type="t20",
            status="In progress", venue="Venue", date="2026-09-26",
            teams=["South Africa", "England"], match_started=True, match_ended=False,
        )
        pred = predict_live_state(features, dummy_match, model=self.model)
        self.assertAlmostEqual(pred.win_probability + pred.loss_probability, 1.0, places=4)

    def test_15_duplicate_state_suppression(self):
        """15. Verify duplicate match state suppresses swing events."""
        tracker = LiveStateTracker()
        pred = LivePredictionResult(
            match_id="m15", match_name="SA vs ENG", chasing_team="SA", defending_team="ENG",
            current_score=148.0, wickets_lost=4, overs_completed=17.0,
            target_score=176.0, runs_remaining=28.0, balls_remaining=18,
            current_run_rate=8.71, required_run_rate=9.33,
            win_probability=0.612, loss_probability=0.388, win_probability_pct=61.2,
        )
        sig = ("m15", 2, "148/4", False, ((("SA", 148, 4, 102),)))
        is_new1, _, _ = tracker.update(pred, sig)
        self.assertTrue(is_new1)
        # Duplicate poll:
        is_new2, chg2, abs2 = tracker.update(pred, sig)
        self.assertFalse(is_new2)
        self.assertEqual(chg2, 0.0)

    def test_16_terminal_target_reached(self):
        """16. Verify terminal state: target reached yields 1.0 win probability."""
        raw = {
            "id": "t16", "name": "SA vs ENG", "matchType": "t20",
            "status": "South Africa won by 6 wickets", "matchEnded": True,
            "teams": ["England", "South Africa"],
            "score": [
                {"r": 175, "w": 6, "o": 20, "inning": "England Inning 1"},
                {"r": 176, "w": 4, "o": 19.1, "inning": "South Africa Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, _ = extract_match_state_features(norm)
        self.assertTrue(features.is_terminal)
        self.assertEqual(features.terminal_probability, 1.0)
        self.assertIn("Target Reached", features.terminal_state)

    def test_17_terminal_all_out(self):
        """17. Verify terminal state: all out yields 0.0 win probability."""
        raw = {
            "id": "t17", "name": "SA vs ENG", "matchType": "t20",
            "status": "England won by 20 runs", "matchEnded": True,
            "teams": ["England", "South Africa"],
            "score": [
                {"r": 175, "w": 6, "o": 20, "inning": "England Inning 1"},
                {"r": 155, "w": 10, "o": 18.4, "inning": "South Africa Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, _ = extract_match_state_features(norm)
        self.assertTrue(features.is_terminal)
        self.assertEqual(features.terminal_probability, 0.0)
        self.assertIn("All Out", features.terminal_state)

    def test_18_terminal_balls_exhausted(self):
        """18. Verify terminal state: balls exhausted yields 0.0 win probability."""
        raw = {
            "id": "t18", "name": "SA vs ENG", "matchType": "t20",
            "status": "England won by 5 runs", "matchEnded": True,
            "teams": ["England", "South Africa"],
            "score": [
                {"r": 175, "w": 6, "o": 20, "inning": "England Inning 1"},
                {"r": 170, "w": 7, "o": 20.0, "inning": "South Africa Inning 1"},
            ],
        }
        norm = normalize_match_dict(raw)
        features, _ = extract_match_state_features(norm)
        self.assertTrue(features.is_terminal)
        self.assertEqual(features.terminal_probability, 0.0)
        self.assertIn("Balls Exhausted", features.terminal_state)

    def test_19_malformed_api_response(self):
        """19. Verify malformed API response handling."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "{not valid json}"
        mock_resp.json.side_effect = ValueError("Invalid JSON syntax")
        with patch("src.live.cricket_api.requests.Session.get", return_value=mock_resp):
            client = CricketApiClient(api_key="valid-dummy-key")
            with self.assertRaises(CricketApiError) as ctx:
                client.get_current_matches()
            self.assertIn("malformed json", str(ctx.exception).lower())

    def test_20_api_failure_handling(self):
        """20. Verify API timeout handling and rate limit handling."""
        import requests
        with patch("src.live.cricket_api.requests.Session.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
            client = CricketApiClient(api_key="valid-dummy-key")
            with self.assertRaises(CricketApiError) as ctx:
                client.get_current_matches()
            self.assertIn("timed out", str(ctx.exception).lower())


class TestMatchStatusNormalization(unittest.TestCase):
    """Tests for distinguishing live vs completed vs upcoming matches (Requirements 1-4)."""

    def test_genuine_live_t20_match(self):
        """1. Genuine live T20 match -> status = live."""
        match_dict = {
            "id": "live-1",
            "name": "Team A vs Team B",
            "matchType": "t20",
            "status": "Team B need 24 runs in 18 balls",
            "matchStarted": True,
            "matchEnded": False,
        }
        self.assertFalse(is_completed_match(match_dict))
        self.assertEqual(normalize_match_status(match_dict), "live")

        norm = normalize_match_dict(match_dict)
        self.assertFalse(is_completed_match(norm))
        self.assertEqual(normalize_match_status(norm), "live")

    def test_completed_t20_match_won_by_wickets(self):
        """2. Completed T20 match with 'Team A won by 4 wickets' -> status = completed."""
        match_dict = {
            "id": "comp-1",
            "name": "Sri Lanka Women vs Pakistan Women",
            "matchType": "t20",
            "status": "Sri Lanka Women won by 4 wickets",
            "matchStarted": True,
            "matchEnded": True,
        }
        self.assertTrue(is_completed_match(match_dict))
        self.assertEqual(normalize_match_status(match_dict), "completed")

        norm = normalize_match_dict(match_dict)
        self.assertTrue(is_completed_match(norm))
        self.assertEqual(normalize_match_status(norm), "completed")

    def test_completed_t20_match_won_by_runs(self):
        """3. Completed T20 match with 'Team A won by 10 runs' -> status = completed."""
        match_dict = {
            "id": "comp-2",
            "name": "India vs Australia",
            "matchType": "t20i",
            "status": "India won by 10 runs",
            "matchStarted": True,
            "matchEnded": False,  # even if provider flag is lagged, status text determines completion
        }
        self.assertTrue(is_completed_match(match_dict))
        self.assertEqual(normalize_match_status(match_dict), "completed")

        norm = normalize_match_dict(match_dict)
        self.assertTrue(is_completed_match(norm))
        self.assertEqual(normalize_match_status(norm), "completed")

    def test_completed_t20_match_tied_no_result_abandoned(self):
        """4. Match tied / no result / abandoned -> status = completed."""
        statuses = [
            "Match tied",
            "Match tied (Super Over won by Team A)",
            "Match abandoned without a ball bowled",
            "No result due to rain",
            "Match drawn",
        ]
        for st in statuses:
            m = {
                "id": f"comp-{st[:4]}",
                "name": "Match X",
                "matchType": "t20",
                "status": st,
                "matchStarted": True,
                "matchEnded": False,
            }
            self.assertTrue(is_completed_match(m), f"Failed for status: {st}")
            self.assertEqual(normalize_match_status(m), "completed", f"Failed for status: {st}")

    def test_upcoming_t20_match(self):
        """Upcoming T20 match (not started) -> status = upcoming."""
        match_dict = {
            "id": "up-1",
            "name": "England vs South Africa",
            "matchType": "t20i",
            "status": "Match starts at 19:00 local time",
            "matchStarted": False,
            "matchEnded": False,
        }
        self.assertFalse(is_completed_match(match_dict))
        self.assertEqual(normalize_match_status(match_dict), "upcoming")


if __name__ == "__main__":
    unittest.main()


