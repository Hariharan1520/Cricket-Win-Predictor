"""
Live Match Tracker and Win Probability Predictor.

Connects to the Cricket Data API, normalizes live match data, feeds match-state
features to the frozen Logistic Regression model, and tracks win probability swings
between consecutive live deliveries / overs.

Usage:
    # List available live T20/T20I matches
    python -m src.live.live_match --list

    # List recent T20/T20I matches
    python -m src.live.live_match --list --recent

    # Run single-state prediction for a specific match
    python -m src.live.live_match --match_id <MATCH_ID> --once

    # Run live polling loop (default interval 30s)
    python -m src.live.live_match --match_id <MATCH_ID> --interval 30
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from dotenv import load_dotenv

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

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/logistic_regression_win_probability.joblib")


@dataclass
class LivePredictionResult:
    match_id: str
    match_name: str
    chasing_team: str
    defending_team: str
    current_score: float
    wickets_lost: int
    overs_completed: float
    target_score: float
    runs_remaining: float
    balls_remaining: int
    current_run_rate: float
    required_run_rate: float
    win_probability: float
    loss_probability: float
    win_probability_pct: float
    overs_bowled_str: str = ""
    is_terminal: bool = False
    terminal_state: Optional[str] = None


class LiveStateTracker:
    """
    In-memory state cache that detects true cricket-state changes and
    tracks Win Probability Swings across live updates.
    """

    def __init__(self):
        self.last_state_signature: Optional[Tuple] = None
        self.last_prediction: Optional[LivePredictionResult] = None
        self.last_update_time: Optional[float] = None

    def get_state_signature(self, norm_match: NormalizedMatch) -> Tuple:
        """
        Creates a hashable tuple representing the core cricket state.
        Ensures updates are triggered ONLY when score, wickets, balls, innings,
        or official match status change.
        """
        scores_sig = tuple(
            (sc.team, sc.runs, sc.wickets, sc.total_legal_balls)
            for sc in norm_match.innings
        )
        return (
            norm_match.match_id,
            len(norm_match.innings),
            norm_match.status,
            norm_match.match_ended,
            scores_sig,
        )

    def update(
        self,
        prediction: LivePredictionResult,
        signature: Tuple,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Updates the tracker state.

        Returns:
            (is_new_state, win_probability_change, absolute_probability_change)
        """
        if self.last_state_signature == signature:
            return False, 0.0, 0.0

        win_prob_change = None
        abs_prob_change = None

        if self.last_prediction is not None:
            win_prob_change = round(
                prediction.win_probability - self.last_prediction.win_probability, 4
            )
            abs_prob_change = round(abs(win_prob_change), 4)

        self.last_state_signature = signature
        self.last_prediction = prediction
        self.last_update_time = time.time()

        return True, win_prob_change, abs_prob_change


def load_prediction_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Loads the pre-trained Logistic Regression model pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {model_path}. "
            "Please ensure Phase 3 model is present."
        )
    return joblib.load(model_path)


def predict_live_state(
    features: LiveMatchFeatures,
    norm_match: NormalizedMatch,
    model=None,
) -> LivePredictionResult:
    """
    Generates win probability from LiveMatchFeatures.
    Handles terminal boundaries deterministically (1.0 or 0.0).
    """
    if features.is_terminal:
        win_prob = features.terminal_probability
        loss_prob = round(1.0 - win_prob, 4)
    else:
        if model is None:
            model = load_prediction_model()
        df = features.to_dataframe()
        probs = model.predict_proba(df)[0]
        loss_prob = round(float(probs[0]), 4)
        win_prob = round(float(probs[1]), 4)

    return LivePredictionResult(
        match_id=norm_match.match_id,
        match_name=norm_match.name,
        chasing_team=features.chasing_team,
        defending_team=features.defending_team,
        current_score=features.current_score,
        wickets_lost=features.wickets_lost,
        overs_completed=features.overs_completed,
        target_score=features.target_score,
        runs_remaining=features.runs_remaining,
        balls_remaining=features.balls_remaining,
        current_run_rate=features.current_run_rate,
        required_run_rate=features.required_run_rate,
        win_probability=win_prob,
        loss_probability=loss_prob,
        win_probability_pct=round(win_prob * 100, 1),
        overs_bowled_str=features.overs_bowled_str,
        is_terminal=features.is_terminal,
        terminal_state=features.terminal_state,
    )


def display_match_list(matches: List[NormalizedMatch], title: str = "T20 LIVE MATCHES"):
    """Prints formatted details of filtered T20/T20I matches."""
    print("\n" + "=" * 65)
    print(title)
    print("-" * 65)
    if not matches:
        print("No live T20/T20I matches are currently available.")
        print("=" * 65 + "\n")
        return

    for idx, m in enumerate(matches):
        fmt = m.match_type.upper() if m.match_type else "T20"
        print(f"[{idx+1}] {m.name}")
        print(f"    Format:   {fmt}")
        print(f"    Status:   {m.status}")
        print(f"    Match ID: {m.match_id}")
        if idx < len(matches) - 1:
            print("-" * 65)
    print("=" * 65 + "\n")


def display_live_prediction(
    prediction: LivePredictionResult,
    norm_match: NormalizedMatch,
    win_prob_change: Optional[float] = None,
    is_cached: bool = False,
):
    """Prints the live win-probability prediction panel."""
    print("\n" + "=" * 65)
    print(f"LIVE WIN PROBABILITY UPDATE: {prediction.match_name}")
    print("=" * 65)
    print(f"Status:       {norm_match.status}")
    print(f"Venue:        {norm_match.venue}")
    print(f"Equation:     {prediction.chasing_team} need {prediction.runs_remaining:.0f} runs from {prediction.balls_remaining} balls")
    overs_display = f"{prediction.overs_bowled_str} ov ({prediction.overs_completed:.2f} fractional)" if prediction.overs_bowled_str else f"{prediction.overs_completed:.1f} ov"
    print(f"Score:        {prediction.current_score:.0f}/{prediction.wickets_lost} in {overs_display} (Target: {prediction.target_score:.0f})")
    print(f"CRR / RRR:    {prediction.current_run_rate:.2f} / {prediction.required_run_rate:.2f}")
    print("-" * 65)

    if prediction.is_terminal:
        print(f"MATCH TERMINAL: {prediction.terminal_state}")
    
    print(f"{prediction.chasing_team} Win Probability: {prediction.win_probability_pct:.1f}%")
    print(f"{prediction.defending_team} Win Probability: {round(prediction.loss_probability * 100, 1):.1f}%")

    if is_cached:
        print("Note: Match state unchanged since last poll (duplicate update skipped).")
    elif win_prob_change is not None:
        sign = "+" if win_prob_change >= 0 else ""
        print(f"Win Probability Swing:        {sign}{win_prob_change*100:.1f} percentage points")
    else:
        print("Win Probability Swing:        Baseline update (first live observation)")
    print("=" * 65 + "\n")


def run_live_tracker(
    client: CricketApiClient,
    match_id: str,
    poll_interval: float = 30.0,
    max_iterations: Optional[int] = None,
    model=None,
):
    """
    Continuously polls the Cricket Data API for the selected match,
    evaluates win probability, and reports probability swings.
    """
    if model is None:
        model = load_prediction_model()

    tracker = LiveStateTracker()
    iteration = 0

    print(f"\n[INFO] Starting live tracker for Match ID: {match_id}")
    print(f"[INFO] Poll interval: {poll_interval}s | Press Ctrl+C to terminate.")

    try:
        while True:
            iteration += 1
            logger.info(f"Polling match {match_id} (Iteration #{iteration})...")
            try:
                raw_data = client.get_match_info(match_id)
            except CricketApiRateLimitError as e:
                print(f"\n[RATE LIMIT] {e}")
                break
            except CricketApiError as e:
                print(f"\n[API ERROR] {e}")
                time.sleep(poll_interval)
                continue

            norm_match = normalize_match_dict(raw_data)
            features, reason = extract_match_state_features(norm_match)

            if features is None:
                print("\n" + "=" * 65)
                print(f"MATCH: {norm_match.name}")
                print(f"Status: {norm_match.status}")
                print(f"[UNAVAILABLE] {reason}")
                print("=" * 65 + "\n")
            else:
                signature = tracker.get_state_signature(norm_match)
                prediction = predict_live_state(features, norm_match, model=model)
                is_new, prob_change, abs_change = tracker.update(prediction, signature)

                display_live_prediction(
                    prediction=prediction,
                    norm_match=norm_match,
                    win_prob_change=prob_change,
                    is_cached=not is_new,
                )

                if features.is_terminal and norm_match.match_ended:
                    print("[INFO] Match has concluded. Live polling terminating.")
                    break

            if max_iterations is not None and iteration >= max_iterations:
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[INFO] Live tracking stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Live Cricket Win Probability Predictor (T20 Only)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available live T20/T20I matches",
    )
    parser.add_argument(
        "--recent",
        action="store_true",
        help="Include recent completed/scheduled T20 matches from /matches endpoint",
    )
    parser.add_argument(
        "--match_id",
        type=str,
        default=None,
        help="API match ID to track",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Polling interval in seconds (default from LIVE_POLL_INTERVAL_SECONDS or 30s)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch state once, predict, and exit immediately without polling",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained Logistic Regression joblib model",
    )

    args = parser.parse_args()

    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    poll_interval = args.interval
    if poll_interval is None:
        try:
            poll_interval = float(os.getenv("LIVE_POLL_INTERVAL_SECONDS", "30"))
        except ValueError:
            poll_interval = 30.0

    try:
        client = CricketApiClient()
    except CricketApiAuthError as e:
        print(f"\n[AUTHENTICATION ERROR] {e}\n")
        sys.exit(1)

    model = load_prediction_model(Path(args.model))

    # Fetch matches if listing or choosing
    if args.list or not args.match_id:
        print("\nFetching matches from Cricket Data API...")
        try:
            if args.recent:
                raw_matches = client.get_matches()
                title = "T20 MATCHES (RECENT / LIVE)"
            else:
                raw_matches = client.get_current_matches()
                title = "T20 LIVE MATCHES"
        except CricketApiError as e:
            print(f"[API ERROR] {e}")
            sys.exit(1)

        all_norm_matches = [normalize_match_dict(m) for m in raw_matches]
        norm_matches = filter_t20_matches(all_norm_matches)

        display_match_list(norm_matches, title=title)

        if not norm_matches:
            sys.exit(0)

        if args.list:
            sys.exit(0)

        # Interactive selection
        try:
            choice = input(
                "Enter match number to track (or 'q' to quit, 'r' for recent T20s): "
            ).strip()
            if choice.lower() == "q":
                sys.exit(0)
            elif choice.lower() == "r":
                raw_recent = client.get_matches()
                norm_matches = filter_t20_matches([normalize_match_dict(m) for m in raw_recent])
                display_match_list(norm_matches, title="T20 MATCHES (RECENT / LIVE)")
                if not norm_matches:
                    sys.exit(0)
                choice = input("Enter match number to track (or 'q' to quit): ").strip()
                if choice.lower() == "q":
                    sys.exit(0)

            idx = int(choice) - 1
            if 0 <= idx < len(norm_matches):
                chosen_match_id = norm_matches[idx].match_id
            else:
                print("Invalid selection.")
                sys.exit(1)
        except (ValueError, EOFError):
            print("Invalid input.")
            sys.exit(1)
    else:
        chosen_match_id = args.match_id

    # Execute tracker
    max_iters = 1 if args.once else None
    run_live_tracker(
        client=client,
        match_id=chosen_match_id,
        poll_interval=poll_interval,
        max_iterations=max_iters,
        model=model,
    )


if __name__ == "__main__":
    main()
