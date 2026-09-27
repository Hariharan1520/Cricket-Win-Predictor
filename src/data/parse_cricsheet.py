"""
Cricsheet T20 JSON Parser
Extracts ball-by-ball match data from raw Cricsheet JSON files into a structured tabular format.

Usage:
    # Process a single match JSON
    python src/data/parse_cricsheet.py --input data/raw/1001349.json

    # Process all matches in data/raw/
    python src/data/parse_cricsheet.py --input data/raw/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_match_json(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Parses a single Cricsheet JSON file into delivery-level records.

    Args:
        file_path: Path to the Cricsheet JSON match file.

    Returns:
        List of delivery row dictionaries.
    """
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    match_id = file_path.stem

    # Extract match metadata
    dates = info.get("dates", [])
    match_date = str(dates[0]) if dates else None
    season = str(info.get("season", "")) if "season" in info else None
    venue = info.get("venue")
    city = info.get("city")
    gender = info.get("gender")
    match_type = info.get("match_type")
    team_type = info.get("team_type")

    # Toss info
    toss = info.get("toss", {})
    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision")

    # Match outcome
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")  # May be None for ties or no-results

    teams = info.get("teams", [])

    rows: List[Dict[str, Any]] = []
    innings_list = data.get("innings", [])

    for inn_idx, innings in enumerate(innings_list, start=1):
        batting_team = innings.get("team")

        # Determine bowling team reliably
        bowling_team = None
        if len(teams) >= 2:
            bowling_team = teams[1] if teams[0] == batting_team else teams[0]

        cumulative_score = 0
        cumulative_wickets = 0

        for over_data in innings.get("overs", []):
            over_number = int(over_data.get("over", 0))

            # Reset legal ball counter at start of every over
            legal_ball_counter = 0

            deliveries = over_data.get("deliveries", [])
            for deliv_idx, deliv in enumerate(deliveries, start=1):
                delivery_number = deliv_idx

                # Batter, bowler, non-striker
                batter = deliv.get("batter")
                non_striker = deliv.get("non_striker")
                bowler = deliv.get("bowler")

                # Runs
                runs_info = deliv.get("runs", {})
                runs_batter = int(runs_info.get("batter", 0))
                runs_extras = int(runs_info.get("extras", 0))
                runs_total = int(runs_info.get("total", 0))

                # Extras handling
                extras_info = deliv.get("extras", {})
                extras_type = ",".join(sorted(extras_info.keys())) if extras_info else None
                is_wide = "wides" in extras_info
                is_noball = "noballs" in extras_info

                # Legal ball calculation
                if is_wide or is_noball:
                    legal_ball = 0
                else:
                    legal_ball = 1
                    legal_ball_counter += 1

                ball_in_over = legal_ball_counter

                # Wickets handling
                wickets_info = deliv.get("wickets", [])
                if wickets_info:
                    wicket = 1
                    w_first = wickets_info[0]
                    wicket_kind = w_first.get("kind")
                    player_out = w_first.get("player_out")
                else:
                    wicket = 0
                    wicket_kind = None
                    player_out = None

                # Cumulative innings state AFTER this delivery
                cumulative_score += runs_total
                cumulative_wickets += wicket

                rows.append({
                    "match_id": match_id,
                    "date": match_date,
                    "season": season,
                    "venue": venue,
                    "city": city,
                    "gender": gender,
                    "match_type": match_type,
                    "team_type": team_type,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,
                    "innings_number": inn_idx,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over_number": over_number,
                    "delivery_number": delivery_number,
                    "legal_ball": legal_ball,
                    "ball_in_over": ball_in_over,
                    "batter": batter,
                    "non_striker": non_striker,
                    "bowler": bowler,
                    "runs_batter": runs_batter,
                    "runs_extras": runs_extras,
                    "runs_total": runs_total,
                    "extras_type": extras_type,
                    "wicket": wicket,
                    "wicket_kind": wicket_kind,
                    "player_out": player_out,
                    "cumulative_score": cumulative_score,
                    "cumulative_wickets": cumulative_wickets,
                    "winner": winner,
                })

    return rows


def parse_all_matches(
    input_path: Union[str, Path],
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Parses a single file or directory of Cricsheet JSON files.

    Args:
        input_path: Path to file or directory.
        limit: Optional maximum number of files to process.

    Returns:
        Pandas DataFrame of ball-by-ball deliveries.
    """
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(list(input_path.glob("*.json")))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if limit is not None and limit > 0:
        files = files[:limit]

    total_files = len(files)
    logger.info(f"Processing {total_files} match file(s)...")

    all_rows: List[Dict[str, Any]] = []
    start_time = time.time()

    for idx, f in enumerate(files, start=1):
        try:
            match_rows = parse_match_json(f)
            all_rows.extend(match_rows)
        except Exception as e:
            logger.warning(f"Error parsing {f.name}: {e}")

        if idx % 1000 == 0 or idx == total_files:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            logger.info(f"Progress: {idx}/{total_files} files parsed ({rate:.1f} files/s)")

    df = pd.DataFrame(all_rows)
    return df


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Performs data validation and outputs summary metrics.

    Args:
        df: Processed ball-by-ball DataFrame.

    Returns:
        True if all critical checks pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("DATASET VALIDATION & SUMMARY METRICS")
    print("=" * 60)

    num_matches = df["match_id"].nunique()
    num_innings = df.groupby(["match_id", "innings_number"]).ngroups
    num_deliveries = len(df)
    num_legal_balls = int(df["legal_ball"].sum())
    num_wickets = int(df["wicket"].sum())
    total_runs = int(df["runs_total"].sum())
    teams_list = sorted(list(set(df["batting_team"].dropna().unique())))
    winners_count = df.groupby("match_id")["winner"].first().dropna().value_counts()

    print(f"Number of matches:     {num_matches:,}")
    print(f"Number of innings:     {num_innings:,}")
    print(f"Number of deliveries:  {num_deliveries:,}")
    print(f"Number of legal balls: {num_legal_balls:,}")
    print(f"Number of wickets:     {num_wickets:,}")
    print(f"Total runs:            {total_runs:,}")
    print(f"Unique teams ({len(teams_list)}):    {teams_list[:10]} ...")
    print(f"Top 5 match winners:\n{winners_count.head(5).to_string()}")
    print(f"Dataset shape:         {df.shape}")

    print("\nFirst 10 rows:")
    cols_to_preview = [
        "match_id",
        "innings_number",
        "over_number",
        "delivery_number",
        "legal_ball",
        "ball_in_over",
        "batting_team",
        "batter",
        "bowler",
        "runs_total",
        "wicket",
        "cumulative_score",
        "cumulative_wickets",
    ]
    print(df[cols_to_preview].head(10).to_string(index=False))

    # Integrity Checks
    print("\n" + "-" * 60)
    print("INTEGRITY CHECKS")
    print("-" * 60)
    passed = True

    # 1. Missing required columns
    required_cols = [
        "match_id",
        "date",
        "innings_number",
        "batting_team",
        "bowling_team",
        "over_number",
        "delivery_number",
        "legal_ball",
        "ball_in_over",
        "batter",
        "bowler",
        "runs_batter",
        "runs_extras",
        "runs_total",
        "wicket",
        "cumulative_score",
        "cumulative_wickets",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[FAIL] Missing required columns: {missing_cols}")
        passed = False
    else:
        print("[PASS] All required columns present.")

    # 2. Negative runs
    neg_runs = (df["runs_batter"] < 0) | (df["runs_extras"] < 0) | (df["runs_total"] < 0)
    if neg_runs.any():
        print(f"[FAIL] Found {neg_runs.sum()} rows with negative runs.")
        passed = False
    else:
        print("[PASS] Zero negative runs found.")

    # 3. Invalid innings numbers (positive integers; accommodates standard innings and multiple super overs)
    invalid_innings = (df["innings_number"] < 1) | (df["innings_number"].isnull())
    if invalid_innings.any():
        print(f"[FAIL] Found {invalid_innings.sum()} rows with invalid innings numbers.")
        passed = False
    else:
        min_inn, max_inn = int(df["innings_number"].min()), int(df["innings_number"].max())
        print(f"[PASS] All innings numbers valid (positive integers: {min_inn} to {max_inn}).")

    # 4. Unexpected null values in mandatory fields
    null_issues = {}
    for col in required_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            null_issues[col] = int(null_count)
    if null_issues:
        print(f"[FAIL] Unexpected null values in required columns: {null_issues}")
        passed = False
    else:
        print("[PASS] No unexpected null values in mandatory columns.")

    # 5. Data types check
    int_cols = [
        "innings_number",
        "over_number",
        "delivery_number",
        "legal_ball",
        "ball_in_over",
        "runs_batter",
        "runs_extras",
        "runs_total",
        "wicket",
        "cumulative_score",
        "cumulative_wickets",
    ]
    dtype_issues = [c for c in int_cols if not pd.api.types.is_integer_dtype(df[c])]
    if dtype_issues:
        print(f"[FAIL] Non-integer type for columns: {dtype_issues}")
        passed = False
    else:
        print("[PASS] All numeric columns have expected integer data types.")

    print("=" * 60 + "\n")
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Parse Cricsheet T20 JSON data to ball-by-ball CSV"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Path to a single Cricsheet JSON file or a directory containing JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/ball_by_ball.csv",
        help="Path where processed CSV will be saved",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of JSON files to process (useful for testing)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = parse_all_matches(input_path, limit=args.limit)

    if df.empty:
        logger.error("No deliveries were parsed. Exiting.")
        sys.exit(1)

    logger.info(f"Saving {len(df):,} deliveries to {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info("Save complete.")

    # Validation
    is_valid = validate_dataframe(df)
    if not is_valid:
        logger.warning("Validation completed with warnings or failures.")
    else:
        logger.info("Validation completed successfully.")


if __name__ == "__main__":
    main()
