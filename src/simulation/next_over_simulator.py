"""
Phase 6: What-If Next-Over Win Probability Simulator
Simulates hypothetical outcomes for the next over of a T20 chase (6 legal balls),
evaluating the model's projected win-probability response across controlled scenarios.

Usage:
    # Manual state simulation (TEST A)
    python src/simulation/next_over_simulator.py --target_score 190 --current_score 142 --wickets_lost 4 --balls_remaining 24 --overs_completed 16.0

    # Real match state simulation
    python src/simulation/next_over_simulator.py --match_id 430885 --row_index 90

    # Run complete automated validation suite (TEST A through TEST E)
    python src/simulation/next_over_simulator.py --run_tests
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
MATCH_STATE_PATH = Path("data/processed/match_state.csv")

# Baseline controlled scenarios (runs, wickets)
BASELINE_SCENARIOS = [
    ("Dot Over", 0, 0),
    ("Low Scoring Over", 4, 0),
    ("Moderate Over", 6, 0),
    ("Good Over", 8, 0),
    ("Strong Over", 10, 0),
    ("Very Good Over", 12, 0),
    ("Excellent Over", 15, 0),
    ("Explosive Over", 18, 0),
    ("6 runs + 1 wicket", 6, 1),
    ("10 runs + 1 wicket", 10, 1),
    ("15 runs + 1 wicket", 15, 1),
]


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Loads the trained Logistic Regression pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def compute_derived_features(
    target_score: float,
    current_score: float,
    balls_remaining: int,
    overs_completed: float,
) -> Tuple[float, float, float]:
    """
    Computes derived match-state features using exact Phase 2 definitions.
    """
    runs_remaining = max(0.0, float(target_score - current_score))

    current_run_rate = (
        round(float(current_score / overs_completed), 4)
        if overs_completed > 0
        else 0.0
    )

    required_run_rate = (
        round(float(runs_remaining / (balls_remaining / 6.0)), 4)
        if balls_remaining > 0 and runs_remaining > 0
        else 0.0
    )

    return runs_remaining, current_run_rate, required_run_rate


def predict_probability(
    model,
    target_score: float,
    current_score: float,
    wickets_lost: int,
    runs_remaining: float,
    balls_remaining: int,
    overs_completed: float,
    current_run_rate: float,
    required_run_rate: float,
) -> float:
    """Evaluates the model pipeline for non-terminal states."""
    row = pd.DataFrame(
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
    prob = float(model.predict_proba(row)[0, 1])
    return round(prob, 4)


def simulate_next_over(
    target_score: float,
    current_score: float,
    wickets_lost: int,
    balls_remaining: int,
    overs_completed: float,
    model=None,
    scenarios: List[Tuple[str, int, int]] = BASELINE_SCENARIOS,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Runs hypothetical next-over scenario simulations from the specified state.
    """
    if model is None:
        model = load_model()

    # Derived current state features
    runs_remaining, current_run_rate, required_run_rate = compute_derived_features(
        target_score=target_score,
        current_score=current_score,
        balls_remaining=balls_remaining,
        overs_completed=overs_completed,
    )

    # Calculate current state probability (with terminal check)
    if runs_remaining <= 0:
        current_win_prob = 1.0
        current_terminal = "Target Reached (1.0)"
    elif (balls_remaining <= 0 and runs_remaining > 0) or wickets_lost >= 10:
        current_win_prob = 0.0
        current_terminal = "Chase Lost / All Out (0.0)"
    else:
        current_win_prob = predict_probability(
            model=model,
            target_score=target_score,
            current_score=current_score,
            wickets_lost=wickets_lost,
            runs_remaining=runs_remaining,
            balls_remaining=balls_remaining,
            overs_completed=overs_completed,
            current_run_rate=current_run_rate,
            required_run_rate=required_run_rate,
        )
        current_terminal = "In Progress"

    current_state_info = {
        "target_score": target_score,
        "current_score": current_score,
        "wickets_lost": wickets_lost,
        "balls_remaining": balls_remaining,
        "overs_completed": overs_completed,
        "runs_remaining": runs_remaining,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
        "current_win_probability": current_win_prob,
        "current_win_probability_pct": round(current_win_prob * 100, 1),
        "terminal_state": current_terminal,
    }

    # Simulate scenarios
    balls_in_scenario = min(6, balls_remaining)
    is_partial_over = balls_remaining < 6 and balls_remaining > 0

    results = []
    for sc_name, sc_runs, sc_wkts in scenarios:
        # Check invalid wicket condition
        if wickets_lost >= 10 and sc_wkts > 0:
            results.append(
                {
                    "scenario_name": sc_name,
                    "scenario_runs": sc_runs,
                    "scenario_wickets": sc_wkts,
                    "current_score": current_score,
                    "current_wickets": wickets_lost,
                    "new_score": current_score,
                    "new_wickets": wickets_lost,
                    "target_score": target_score,
                    "runs_remaining": runs_remaining,
                    "balls_remaining": balls_remaining,
                    "overs_completed": overs_completed,
                    "current_win_probability": current_win_prob,
                    "simulated_win_probability": np.nan,
                    "probability_change": np.nan,
                    "terminal_state": "All Out",
                    "status": "Invalid (Team already 10 down)",
                }
            )
            continue

        new_score = current_score + sc_runs
        new_wkts = min(10, wickets_lost + sc_wkts)
        new_balls = max(0, balls_remaining - 6)
        new_overs = overs_completed + (balls_in_scenario / 6.0)

        new_runs_rem, new_crr, new_rrr = compute_derived_features(
            target_score=target_score,
            current_score=new_score,
            balls_remaining=new_balls,
            overs_completed=new_overs,
        )

        # Terminal handling
        is_term = False
        term_desc = "Non-Terminal"
        if new_score >= target_score:
            sim_prob = 1.0
            is_term = True
            term_desc = "Target Reached (1.0)"
        elif new_balls <= 0 and new_score < target_score:
            sim_prob = 0.0
            is_term = True
            term_desc = "Balls Exhausted (0.0)"
        elif new_wkts >= 10 and new_score < target_score:
            sim_prob = 0.0
            is_term = True
            term_desc = "All Out (0.0)"
        else:
            sim_prob = predict_probability(
                model=model,
                target_score=target_score,
                current_score=new_score,
                wickets_lost=new_wkts,
                runs_remaining=new_runs_rem,
                balls_remaining=new_balls,
                overs_completed=new_overs,
                current_run_rate=new_crr,
                required_run_rate=new_rrr,
            )

        prob_change = round(sim_prob - current_win_prob, 4)
        status_label = "Final partial over" if is_partial_over else "Valid"

        results.append(
            {
                "scenario_name": sc_name,
                "scenario_runs": sc_runs,
                "scenario_wickets": sc_wkts,
                "current_score": current_score,
                "current_wickets": wickets_lost,
                "new_score": new_score,
                "new_wickets": new_wkts,
                "target_score": target_score,
                "runs_remaining": new_runs_rem,
                "balls_remaining": new_balls,
                "overs_completed": round(new_overs, 4),
                "current_win_probability": current_win_prob,
                "simulated_win_probability": sim_prob,
                "probability_change": prob_change,
                "terminal_state": term_desc,
                "status": status_label,
            }
        )

    scenarios_df = pd.DataFrame(results)
    return current_state_info, scenarios_df


def print_simulation_table(current_state: Dict[str, Any], df: pd.DataFrame):
    """Prints a clean CLI simulation table."""
    print("\n" + "=" * 65)
    print("CURRENT CHASE STATE")
    print("=" * 65)
    print(f"Target Score:             {current_state['target_score']:.0f}")
    print(f"Current Score:            {current_state['current_score']:.0f}/{current_state['wickets_lost']} in {current_state['overs_completed']:.1f} ov")
    print(f"Equation:                 Need {current_state['runs_remaining']:.0f} runs from {current_state['balls_remaining']} balls")
    print(f"Current Run Rate (CRR):   {current_state['current_run_rate']:.2f}")
    print(f"Required Run Rate (RRR):  {current_state['required_run_rate']:.2f}")
    print(f"Current Win Probability:  {current_state['current_win_probability_pct']:.1f}%")
    print("=" * 65)

    print("\n" + "-" * 75)
    print("NEXT-OVER WHAT-IF SCENARIOS (6 LEGAL DELIVERIES)")
    print("-" * 75)
    print(f"{'Scenario':<20} | {'Runs':<4} | {'Wkts':<4} | {'New Score':<9} | {'Win Prob':<9} | {'Change':<8} | {'Terminal State'}")
    print("-" * 75)
    for _, row in df.iterrows():
        if pd.isna(row["simulated_win_probability"]):
            print(f"{row['scenario_name']:<20} | {row['scenario_runs']:<4} | {row['scenario_wickets']:<4} | {row['new_score']:.0f}/{row['new_wickets']:<6} | {'INVALID':<9} | {'-':<8} | {row['status']}")
        else:
            win_pct = f"{row['simulated_win_probability']*100:.1f}%"
            chg_pct = f"{row['probability_change']*100:+.1f}%"
            print(f"{row['scenario_name']:<20} | {row['scenario_runs']:<4} | {row['scenario_wickets']:<4} | {row['new_score']:.0f}/{row['new_wickets']:<6} | {win_pct:<9} | {chg_pct:<8} | {row['terminal_state']}")
    print("-" * 75 + "\n")


def plot_scenario_chart(current_state: Dict[str, Any], df: pd.DataFrame, output_path: Path):
    """
    Creates a publication-quality horizontal bar chart of simulated win probabilities.
    """
    valid_df = df[df["simulated_win_probability"].notna()].copy()
    valid_df = valid_df.iloc[::-1].reset_index(drop=True)  # Reverse for top-to-bottom bar chart

    scenarios = valid_df["scenario_name"].values
    probs_pct = valid_df["simulated_win_probability"].values * 100
    changes_pct = valid_df["probability_change"].values * 100
    cur_pct = current_state["current_win_probability_pct"]

    # Color code bars based on positive vs negative change
    colors = ["#2ca02c" if chg >= 0 else "#d62728" for chg in changes_pct]

    plt.figure(figsize=(9, 6))
    bars = plt.barh(scenarios, probs_pct, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)

    # Current win probability reference line
    plt.axvline(cur_pct, color="#1f77b4", linestyle="--", linewidth=1.8, label=f"Current Win Prob: {cur_pct:.1f}%")

    # Add text labels on bars
    for bar, prob, chg in zip(bars, probs_pct, changes_pct):
        sign = "+" if chg >= 0 else ""
        label = f" {prob:.1f}% ({sign}{chg:.1f}%)"
        plt.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.title(
        f"What-If Next-Over Scenario Simulator\nMatch State: {current_state['current_score']:.0f}/{current_state['wickets_lost']} in {current_state['overs_completed']:.1f} ov | Target: {current_state['target_score']:.0f} (Need {current_state['runs_remaining']:.0f} off {current_state['balls_remaining']}b)",
        fontsize=12,
        pad=14,
    )
    plt.xlabel("Simulated Chasing-Team Win Probability (%)", fontsize=11)
    plt.xlim(0, 115)
    plt.legend(loc="lower right", fontsize=10.5)
    plt.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved scenario chart to {output_path}")


def run_automated_test_suite(model) -> bool:
    """
    Executes and validates TEST A through TEST E as required by Step 16.
    """
    print("\n" + "=" * 65)
    print("RUNNING AUTOMATED TEST SUITE (TEST A through TEST E)")
    print("=" * 65)
    suite_passed = True

    # TEST A — Normal mid-innings state
    print("\n[TEST A] Normal mid-innings state: 142/4, Target 190, 24 balls left, 16.0 ov")
    cur_a, df_a = simulate_next_over(190, 142, 4, 24, 16.0, model=model)
    assert 0.0 <= cur_a["current_win_probability"] <= 1.0
    assert len(df_a) == 11
    assert (df_a["runs_remaining"] >= 0).all()
    assert (df_a["balls_remaining"] >= 0).all()
    print("  -> Passed TEST A constraints.")

    # TEST B — Chase close to completion
    print("\n[TEST B] Chase close to completion: 142/3, Target 150, 18 balls left, 17.0 ov (Need 8)")
    cur_b, df_b = simulate_next_over(150, 142, 3, 18, 17.0, model=model)
    # Scenarios scoring >= 8 runs should hit target_score >= 150 and have prob 1.0
    target_hits = df_b[df_b["scenario_runs"] >= 8]
    assert (target_hits["simulated_win_probability"] == 1.0).all(), "Target hit did not receive 1.0 prob!"
    assert (target_hits["terminal_state"] == "Target Reached (1.0)").all()
    print(f"  -> Passed TEST B: all {len(target_hits)} scenarios reaching target received 1.0 probability.")

    # TEST C — Final-over state (fewer than 6 balls remaining)
    print("\n[TEST C] Final-over state: 160/5, Target 170, 4 balls left, 19.2 ov (Need 10)")
    cur_c, df_c = simulate_next_over(170, 160, 5, 4, 19.3333, model=model)
    assert (df_c["balls_remaining"] == 0).all(), "Balls remaining in final over should be 0 after next over!"
    assert (df_c["status"] == "Final partial over").all()
    # Scenarios scoring < 10 runs should have 0.0 probability because balls ran out
    failed_chases = df_c[df_c["scenario_runs"] < 10]
    assert (failed_chases["simulated_win_probability"] == 0.0).all(), "Failed chase on ball exhaustion did not receive 0.0!"
    print(f"  -> Passed TEST C: partial over handled with correct ball exhaustion terminal logic.")

    # TEST D — Near all-out state (9 wickets lost)
    print("\n[TEST D] Near all-out state: 140/9, Target 160, 24 balls left, 16.0 ov")
    cur_d, df_d = simulate_next_over(160, 140, 9, 24, 16.0, model=model)
    # Wicket scenarios should result in 10 wickets lost and terminal 0.0
    wkt_scenarios = df_d[df_d["scenario_wickets"] > 0]
    assert (wkt_scenarios["new_wickets"] == 10).all()
    assert (wkt_scenarios["simulated_win_probability"] == 0.0).all()
    assert (wkt_scenarios["terminal_state"] == "All Out (0.0)").all()
    print(f"  -> Passed TEST D: all {len(wkt_scenarios)} wicket scenarios correctly mapped to All Out (0.0).")

    # TEST E — Already-completed target scenario
    print("\n[TEST E] Already-completed target scenario: 152/4, Target 150")
    cur_e, df_e = simulate_next_over(150, 152, 4, 12, 18.0, model=model)
    assert cur_e["current_win_probability"] == 1.0
    assert cur_e["terminal_state"] == "Target Reached (1.0)"
    print("  -> Passed TEST E: completed chase immediately recognized as terminal 1.0.")

    print("\n" + "=" * 65)
    print("ALL 5 AUTOMATED TESTS (A THROUGH E) PASSED SUCCESSFULLY!")
    print("=" * 65 + "\n")
    return suite_passed


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: What-If Next-Over Win Probability Simulator"
    )
    parser.add_argument("--target_score", type=float, default=190.0, help="Target score (Innings 1 final score + 1)")
    parser.add_argument("--current_score", type=float, default=142.0, help="Current runs scored in chase")
    parser.add_argument("--wickets_lost", type=int, default=4, help="Current wickets lost (0 to 10)")
    parser.add_argument("--balls_remaining", type=int, default=24, help="Current legal balls remaining")
    parser.add_argument("--overs_completed", type=float, default=16.0, help="Current overs completed (e.g. 16.0)")
    parser.add_argument("--match_id", type=int, default=None, help="Optional: load match state from match_state.csv by match_id")
    parser.add_argument("--row_index", type=int, default=None, help="Optional: specific row index within match_id")
    parser.add_argument("--output_csv", type=str, default="outputs/next_over_scenarios.csv", help="Path to save output CSV")
    parser.add_argument("--output_chart", type=str, default="outputs/next_over_scenario_chart.png", help="Path to save scenario chart")
    parser.add_argument("--run_tests", action="store_true", help="Run automated test suite (TEST A through TEST E)")

    args = parser.parse_args()

    model = load_model()

    # If run_tests flag is set
    if args.run_tests:
        run_automated_test_suite(model)

    # State extraction (either from match_state.csv or CLI args)
    if args.match_id is not None:
        logger.info(f"Extracting real state for match_id {args.match_id} from {MATCH_STATE_PATH}...")
        ms_df = pd.read_csv(MATCH_STATE_PATH, low_memory=False)
        m_rows = ms_df[ms_df["match_id"] == args.match_id].copy().reset_index(drop=True)
        if len(m_rows) == 0:
            raise ValueError(f"Match ID {args.match_id} not found in {MATCH_STATE_PATH}")

        if args.row_index is not None:
            row_idx = min(args.row_index, len(m_rows) - 1)
        else:
            # Default to ~75% through the match or 16th over
            row_idx = min(int(len(m_rows) * 0.75), len(m_rows) - 1)

        sel_row = m_rows.iloc[row_idx]
        target_score = float(sel_row["target_score"])
        current_score = float(sel_row["current_score"])
        wickets_lost = int(sel_row["wickets_lost"])
        balls_remaining = int(sel_row["balls_remaining"])
        overs_completed = float(sel_row["overs_completed"])
        batting_team = sel_row["batting_team"]
        bowling_team = sel_row["bowling_team"]
        print(f"\n[MATCH STATE LOADED] Match {args.match_id}: {batting_team} chasing vs {bowling_team} (Row {row_idx}/{len(m_rows)})")
    else:
        target_score = args.target_score
        current_score = args.current_score
        wickets_lost = args.wickets_lost
        balls_remaining = args.balls_remaining
        overs_completed = args.overs_completed

    # Run Simulation
    current_state, scenarios_df = simulate_next_over(
        target_score=target_score,
        current_score=current_score,
        wickets_lost=wickets_lost,
        balls_remaining=balls_remaining,
        overs_completed=overs_completed,
        model=model,
    )

    # Print Table
    print_simulation_table(current_state, scenarios_df)

    # Save CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scenarios_df.to_csv(out_csv, index=False)
    logger.info(f"Saved next-over scenarios to {out_csv}")

    # Plot Chart
    out_chart = Path(args.output_chart)
    plot_scenario_chart(current_state, scenarios_df, out_chart)

    # Print summary statistics
    valid_scenarios = scenarios_df[scenarios_df["simulated_win_probability"].notna()]
    min_prob = valid_scenarios["simulated_win_probability"].min()
    max_prob = valid_scenarios["simulated_win_probability"].max()
    max_pos = valid_scenarios["probability_change"].max()
    max_neg = valid_scenarios["probability_change"].min()
    num_term = (valid_scenarios["terminal_state"] != "Non-Terminal").sum()

    print("\n" + "=" * 65)
    print("PHASE 6 SIMULATION METRICS")
    print("=" * 65)
    print(f"Current Win Probability:         {current_state['current_win_probability_pct']:.1f}%")
    print(f"Total Scenarios Generated:       {len(scenarios_df)}")
    print(f"Valid Scenarios:                 {len(valid_scenarios)}")
    print(f"Terminal Scenarios:              {num_term}")
    print(f"Probability Range:               {min_prob*100:.1f}% to {max_prob*100:.1f}%")
    print(f"Largest Positive Scenario Gain:  {max_pos*100:+.1f}% ({valid_scenarios.loc[valid_scenarios['probability_change'].idxmax(), 'scenario_name']})")
    print(f"Largest Negative Scenario Drop:  {max_neg*100:+.1f}% ({valid_scenarios.loc[valid_scenarios['probability_change'].idxmin(), 'scenario_name']})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
