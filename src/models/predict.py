"""
Inference script for T20 Chase Win-Probability Model.

Usage:
    python src/models/predict.py --target_score 180 --current_score 120 --wickets_lost 4 --runs_remaining 60 --balls_remaining 36 --overs_completed 14.0 --current_run_rate 8.57 --required_run_rate 10.0
"""

import argparse
from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd

FEATURE_COLS = [
    "target_score",
    "current_score",
    "wickets_lost",
    "runs_remaining",
    "balls_remaining",
    "overs_completed",
    "current_run_rate",
    "required_run_rate",
]

DEFAULT_MODEL_PATH = Path("models/logistic_regression_win_probability.joblib")


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Loads the trained sklearn pipeline."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Please run train.py first."
        )
    return joblib.load(model_path)


def predict_win_probability(
    target_score: float,
    current_score: float,
    wickets_lost: int,
    runs_remaining: float,
    balls_remaining: int,
    overs_completed: float,
    current_run_rate: float,
    required_run_rate: float,
    model=None,
) -> Dict[str, float]:
    """
    Predicts the win probability for chasing and defending teams.

    Returns:
        Dict with 'chasing_team_win_prob' and 'bowling_team_win_prob' (as decimals 0.0 to 1.0).
    """
    if model is None:
        model = load_model()

    input_df = pd.DataFrame(
        [
            {
                "target_score": target_score,
                "current_score": current_score,
                "wickets_lost": wickets_lost,
                "runs_remaining": runs_remaining,
                "balls_remaining": balls_remaining,
                "overs_completed": overs_completed,
                "current_run_rate": current_run_rate,
                "required_run_rate": required_run_rate,
            }
        ]
    )[FEATURE_COLS]

    probs = model.predict_proba(input_df)[0]
    bowling_team_prob = float(probs[0])
    chasing_team_prob = float(probs[1])

    return {
        "chasing_team_win_prob": chasing_team_prob,
        "bowling_team_win_prob": bowling_team_prob,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict T20 Chase Win Probability using Baseline Logistic Regression"
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to saved joblib model")
    parser.add_argument("--target_score", type=float, required=True, help="Target score set by 1st innings + 1")
    parser.add_argument("--current_score", type=float, required=True, help="Current runs scored by chasing team")
    parser.add_argument("--wickets_lost", type=int, required=True, help="Wickets lost by chasing team (0 to 10)")
    parser.add_argument("--runs_remaining", type=float, required=True, help="Runs remaining to win")
    parser.add_argument("--balls_remaining", type=int, required=True, help="Legal balls remaining in innings")
    parser.add_argument("--overs_completed", type=float, required=True, help="Completed overs (e.g. 5.1667)")
    parser.add_argument("--current_run_rate", type=float, required=True, help="Current run rate (CRR)")
    parser.add_argument("--required_run_rate", type=float, required=True, help="Required run rate (RRR)")

    args = parser.parse_args()

    model = load_model(args.model)
    res = predict_win_probability(
        target_score=args.target_score,
        current_score=args.current_score,
        wickets_lost=args.wickets_lost,
        runs_remaining=args.runs_remaining,
        balls_remaining=args.balls_remaining,
        overs_completed=args.overs_completed,
        current_run_rate=args.current_run_rate,
        required_run_rate=args.required_run_rate,
        model=model,
    )

    chasing_pct = res["chasing_team_win_prob"] * 100
    bowling_pct = res["bowling_team_win_prob"] * 100

    print("\n" + "=" * 45)
    print("T20 CHASE WIN PROBABILITY PREDICTION")
    print("=" * 45)
    print(f"Target Score:      {args.target_score:.0f}")
    print(f"Current State:     {args.current_score:.0f}/{args.wickets_lost} in {args.overs_completed:.1f} ov")
    print(f"Equation:          Need {args.runs_remaining:.0f} runs from {args.balls_remaining} balls")
    print(f"CRR / RRR:         {args.current_run_rate:.2f} / {args.required_run_rate:.2f}")
    print("-" * 45)
    print(f"Chasing team win probability: {chasing_pct:.1f}%")
    print(f"Bowling team win probability: {bowling_pct:.1f}%")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
