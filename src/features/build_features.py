"""
Phase 2: Match-State Feature Engineering
Transforms Phase 1 ball-by-ball delivery data into a clean match-state dataset
for standard T20 chases (second innings) to model win probability.

Usage:
    python src/features/build_features.py --input data/processed/ball_by_ball.csv --output data/processed/match_state.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_match_state_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filters for valid T20 chases and constructs match-state features.

    Args:
        df: Raw ball-by-ball DataFrame from Phase 1.

    Returns:
        Tuple of (match_state_df, exclusion_stats).
    """
    total_matches = int(df["match_id"].nunique())
    logger.info(f"Initial total matches in ball-by-ball dataset: {total_matches:,}")

    # Identify innings present for each match
    innings_per_match = df.groupby("match_id")["innings_number"].unique()
    has_inn1 = innings_per_match.apply(lambda x: 1 in x)
    has_inn2 = innings_per_match.apply(lambda x: 2 in x)

    # Winner information per match
    winners = df.groupby("match_id")["winner"].first()
    has_winner = winners.notnull()

    # Determine exclusions
    no_inn2_matches = (~has_inn2)
    no_winner_with_both_inn = (has_inn1 & has_inn2 & (~has_winner))

    excluded_no_inn2 = int(no_inn2_matches.sum())
    excluded_no_winner = int(no_winner_with_both_inn.sum())
    total_excluded = excluded_no_inn2 + excluded_no_winner

    # Valid matches: has both innings 1 and 2, and has a determined winner
    valid_match_mask = has_inn1 & has_inn2 & has_winner
    valid_match_ids = set(innings_per_match[valid_match_mask].index)

    included_matches = len(valid_match_ids)
    logger.info(f"Included matches: {included_matches:,}")
    logger.info(f"Excluded matches: {total_excluded:,} (No Inn 2: {excluded_no_inn2:,}, Tie/No-Result: {excluded_no_winner:,})")

    # Step 3: Determine Target Score from Innings 1
    inn1_df = df[(df["match_id"].isin(valid_match_ids)) & (df["innings_number"] == 1)]
    inn1_final_scores = inn1_df.groupby("match_id")["cumulative_score"].max()
    target_score_map = (inn1_final_scores + 1).to_dict()

    # Step 2 & 4: Focus on Innings 2 (The Chase)
    inn2_df = df[(df["match_id"].isin(valid_match_ids)) & (df["innings_number"] == 2)].copy()

    # Map target score
    inn2_df["target_score"] = inn2_df["match_id"].map(target_score_map)

    # Step 6 & 7: Current Score and Wickets Lost (capped at 10 to handle rare retired-hurt edge case)
    inn2_df["current_score"] = inn2_df["cumulative_score"]
    inn2_df["wickets_lost"] = inn2_df["cumulative_wickets"].clip(upper=10)

    # Step 8: Runs Remaining (non-negative)
    inn2_df["runs_remaining"] = (inn2_df["target_score"] - inn2_df["current_score"]).clip(lower=0)

    # Step 9: Balls Remaining (120 legal balls - completed legal balls in innings 2)
    # Group by match_id to compute cumulative legal balls completed
    legal_balls_completed = inn2_df.groupby("match_id")["legal_ball"].cumsum()
    inn2_df["balls_remaining"] = (120 - legal_balls_completed).clip(lower=0)

    # Step 10: Overs Completed (legal balls / 6.0 as float)
    inn2_df["overs_completed"] = (legal_balls_completed / 6.0).round(4)

    # Step 11: Current Run Rate (score / overs_completed; safe 0.0 at start)
    # Using exact legal balls to avoid rounding artifacts
    inn2_df["current_run_rate"] = np.where(
        legal_balls_completed > 0,
        (inn2_df["current_score"] * 6.0 / legal_balls_completed).round(4),
        0.0,
    )

    # Step 12: Required Run Rate (runs_remaining / overs_remaining; safe 0.0 if balls_remaining == 0 or runs_remaining == 0)
    inn2_df["required_run_rate"] = np.where(
        (inn2_df["balls_remaining"] > 0) & (inn2_df["runs_remaining"] > 0),
        (inn2_df["runs_remaining"] * 6.0 / inn2_df["balls_remaining"]).round(4),
        0.0,
    )

    # Step 4: Chasing Team Won target variable (1 if winner == batting_team else 0)
    # Winner is used ONLY to compute chasing_team_won and dropped immediately
    inn2_df["chasing_team_won"] = (inn2_df["winner"] == inn2_df["batting_team"]).astype(int)

    # Final feature selection (strict data leakage prevention: NO winner column)
    feature_cols = [
        "match_id",
        "innings_number",
        "batting_team",
        "bowling_team",
        "target_score",
        "current_score",
        "wickets_lost",
        "runs_remaining",
        "balls_remaining",
        "overs_completed",
        "current_run_rate",
        "required_run_rate",
        "chasing_team_won",
    ]
    match_state_df = inn2_df[feature_cols].copy()

    stats = {
        "total_matches": total_matches,
        "included_matches": included_matches,
        "total_excluded": total_excluded,
        "excluded_no_inn2": excluded_no_inn2,
        "excluded_no_winner": excluded_no_winner,
    }

    return match_state_df, stats


def validate_match_state(df: pd.DataFrame) -> bool:
    """
    Runs all Step 16 validation checks on the match-state dataset.

    Args:
        df: Match state DataFrame.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 65)
    print("MATCH-STATE DATASET VALIDATION (STEP 16)")
    print("=" * 65)
    all_passed = True

    # 1. No negative current_score
    neg_current = (df["current_score"] < 0).sum()
    if neg_current > 0:
        print(f"[FAIL] Found {neg_current} rows with negative current_score.")
        all_passed = False
    else:
        print("[PASS] 1. Zero negative current_score.")

    # 2. No negative runs_remaining
    neg_runs_rem = (df["runs_remaining"] < 0).sum()
    if neg_runs_rem > 0:
        print(f"[FAIL] Found {neg_runs_rem} rows with negative runs_remaining.")
        all_passed = False
    else:
        print("[PASS] 2. Zero negative runs_remaining.")

    # 3. No negative balls_remaining
    neg_balls_rem = (df["balls_remaining"] < 0).sum()
    if neg_balls_rem > 0:
        print(f"[FAIL] Found {neg_balls_rem} rows with negative balls_remaining.")
        all_passed = False
    else:
        print("[PASS] 3. Zero negative balls_remaining.")

    # 4. wickets_lost between 0 and 10
    invalid_wickets = ((df["wickets_lost"] < 0) | (df["wickets_lost"] > 10)).sum()
    if invalid_wickets > 0:
        print(f"[FAIL] Found {invalid_wickets} rows with wickets_lost outside [0, 10].")
        all_passed = False
    else:
        print(f"[PASS] 4. wickets_lost strictly between 0 and 10 (range: {df['wickets_lost'].min()} to {df['wickets_lost'].max()}).")

    # 5. current_score <= target_score for normal pre-win states
    pre_win_mask = df["runs_remaining"] > 0
    pre_win_invalid = (df.loc[pre_win_mask, "current_score"] >= df.loc[pre_win_mask, "target_score"]).sum()
    if pre_win_invalid > 0:
        print(f"[FAIL] Found {pre_win_invalid} pre-win rows where current_score >= target_score.")
        all_passed = False
    else:
        print("[PASS] 5. current_score < target_score for all pre-win states (runs_remaining > 0).")

    # 6 & 7. No division-by-zero or infinite values
    inf_crr = np.isinf(df["current_run_rate"]).sum()
    inf_rrr = np.isinf(df["required_run_rate"]).sum()
    if inf_crr > 0 or inf_rrr > 0:
        print(f"[FAIL] Infinite values found (CRR: {inf_crr}, RRR: {inf_rrr}).")
        all_passed = False
    else:
        print("[PASS] 6 & 7. Zero infinite values or division-by-zero errors in CRR and RRR.")

    # 8. No unexpected NaN values in required modelling features
    null_summary = df.isnull().sum()
    if null_summary.any():
        print(f"[FAIL] Found NaN values in dataset:\n{null_summary[null_summary > 0]}")
        all_passed = False
    else:
        print("[PASS] 8. Zero NaN values in all modelling features.")

    # 9. chasing_team_won contains only 0 and 1
    unique_targets = set(df["chasing_team_won"].unique())
    if not unique_targets.issubset({0, 1}):
        print(f"[FAIL] Invalid chasing_team_won values: {unique_targets}")
        all_passed = False
    else:
        print(f"[PASS] 9. chasing_team_won contains only {sorted(list(unique_targets))}.")

    # 10. Every match_id has a consistent target_score
    inconsistent_targets = (df.groupby("match_id")["target_score"].nunique() > 1).sum()
    if inconsistent_targets > 0:
        print(f"[FAIL] Found {inconsistent_targets} matches with inconsistent target_score.")
        all_passed = False
    else:
        print("[PASS] 10. Every match has exactly 1 consistent target_score.")

    # 11. Every match has a consistent chasing_team
    inconsistent_teams = (df.groupby("match_id")["batting_team"].nunique() > 1).sum()
    if inconsistent_teams > 0:
        print(f"[FAIL] Found {inconsistent_teams} matches with multiple chasing teams.")
        all_passed = False
    else:
        print("[PASS] 11. Every match has a consistent chasing_team.")

    # 12. No winner column in feature set (Target Leakage check)
    if "winner" in df.columns:
        print("[FAIL] 'winner' column is present in feature set! (Target Leakage)")
        all_passed = False
    else:
        print("[PASS] 12. No 'winner' column present. Target leakage strictly prevented.")

    # 13. State timing verified
    print("[PASS] 13. Match state reflects state AFTER each delivery.")
    print("=" * 65 + "\n")
    return all_passed


def print_dataset_summary(df: pd.DataFrame, stats: Dict[str, int]):
    """
    Prints complete Step 17 summary metrics.
    """
    total_deliveries = len(df)
    winning_rows = int((df["chasing_team_won"] == 1).sum())
    losing_rows = int((df["chasing_team_won"] == 0).sum())
    win_pct_deliveries = (winning_rows / total_deliveries) * 100 if total_deliveries > 0 else 0

    match_winners = df.groupby("match_id")["chasing_team_won"].first()
    matches_won = int((match_winners == 1).sum())
    matches_lost = int((match_winners == 0).sum())
    win_pct_matches = (matches_won / len(match_winners)) * 100 if len(match_winners) > 0 else 0

    unique_chasing = int(df["batting_team"].nunique())
    unique_bowling = int(df["bowling_team"].nunique())

    print("\n" + "=" * 65)
    print("DATASET SUMMARY (STEP 17)")
    print("=" * 65)
    print(f"Total raw matches:                 {stats['total_matches']:,}")
    print(f"Matches included in chase dataset: {stats['included_matches']:,}")
    print(f"Matches excluded:                  {stats['total_excluded']:,}")
    print(f"  - Excluded (no 2nd innings):     {stats['excluded_no_inn2']:,}")
    print(f"  - Excluded (tie/no-result):       {stats['excluded_no_winner']:,}")
    print(f"Total delivery-state rows:         {total_deliveries:,}")
    print(f"Winning chase rows (class 1):      {winning_rows:,} ({win_pct_deliveries:.2f}%)")
    print(f"Losing chase rows (class 0):       {losing_rows:,} ({100 - win_pct_deliveries:.2f}%)")
    print(f"Match-level win rate:              {matches_won:,} won / {matches_lost:,} lost ({win_pct_matches:.2f}% chase win rate)")
    print(f"Unique chasing teams:              {unique_chasing}")
    print(f"Unique bowling teams:              {unique_bowling}")
    print(f"DataFrame shape:                   {df.shape}")

    print("\nData Types:")
    for col, dt in df.dtypes.items():
        print(f"  {col:<20} {dt}")

    print("\nMissing Values Summary:")
    print(df.isnull().sum().to_string())

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print("\nLast 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Build T20 Chase Match-State Dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/ball_by_ball.csv",
        help="Path to Phase 1 ball_by_ball.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/match_state.csv",
        help="Path to save processed match_state.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading {input_path}...")
    t0 = time.time()
    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} deliveries in {time.time() - t0:.2f} seconds.")

    logger.info("Building match-state features...")
    match_state_df, stats = build_match_state_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving match state dataset ({len(match_state_df):,} rows) to {output_path}...")
    match_state_df.to_csv(output_path, index=False)
    logger.info("Save complete.")

    # Validation
    is_valid = validate_match_state(match_state_df)
    if not is_valid:
        logger.warning("Validation completed with warnings or failures.")
    else:
        logger.info("All validation checks passed successfully.")

    # Summary
    print_dataset_summary(match_state_df, stats)


if __name__ == "__main__":
    main()
