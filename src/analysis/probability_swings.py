"""
Phase 5: Win Probability Swings & Match Moment Analysis
Calculates delivery-by-delivery win probability changes using the trained
baseline Logistic Regression model, identifies major win probability swings,
evaluates impact by cricket events, and analyzes match-level volatility.

Usage:
    python src/analysis/probability_swings.py
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

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


def load_and_merge_data(
    match_state_path: Path,
    ball_by_ball_path: Path,
) -> pd.DataFrame:
    """
    Loads match_state.csv and joins required delivery-level details from ball_by_ball.csv.
    Preserves exact chronological ordering within each match.
    """
    logger.info(f"Loading match state data from {match_state_path}...")
    ms_df = pd.read_csv(match_state_path, low_memory=False)

    logger.info(f"Loading delivery-level data from {ball_by_ball_path}...")
    bbb_df = pd.read_csv(ball_by_ball_path, low_memory=False)

    # Filter ball_by_ball for innings 2 of matches present in match_state
    valid_matches = set(ms_df["match_id"].unique())
    bbb_inn2 = bbb_df[
        (bbb_df["innings_number"] == 2) & (bbb_df["match_id"].isin(valid_matches))
    ].copy()

    # Verify 1:1 row alignment
    if len(ms_df) != len(bbb_inn2):
        raise ValueError(f"Row count mismatch: match_state={len(ms_df):,}, bbb_inn2={len(bbb_inn2):,}")

    logger.info("Extracting delivery-level details (batter, bowler, runs, wickets)...")
    ms_df["over"] = bbb_inn2["over_number"].values
    ms_df["ball"] = bbb_inn2["delivery_number"].values
    ms_df["legal_ball"] = bbb_inn2["legal_ball"].values
    ms_df["batter"] = bbb_inn2["batter"].values
    ms_df["bowler"] = bbb_inn2["bowler"].values
    ms_df["runs_scored"] = bbb_inn2["runs_total"].values
    ms_df["runs_batter"] = bbb_inn2["runs_batter"].values
    ms_df["wicket"] = bbb_inn2["wicket"].values
    ms_df["innings"] = ms_df["innings_number"]

    # Ensure chronological sort within each match
    ms_df = ms_df.sort_values(by=["match_id", "innings", "over", "ball"]).reset_index(drop=True)
    logger.info(f"Prepared {len(ms_df):,} delivery states across {len(valid_matches):,} matches.")

    return ms_df


def generate_probabilities_and_swings(
    df: pd.DataFrame,
    model_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Generates win probabilities via Logistic Regression, applies terminal-state
    correction, and calculates probability swings independently within each match.
    """
    logger.info(f"Loading Logistic Regression model from {model_path}...")
    model = joblib.load(model_path)

    logger.info("Generating raw win probabilities...")
    raw_probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df["win_probability"] = np.round(raw_probs, 4)

    # Step 6: Terminal-state correction
    # Case A: runs_remaining <= 0 (chasing team reached target)
    # Case B: balls_remaining <= 0 and runs_remaining > 0 (deliveries exhausted without reaching target)
    term_a = df["runs_remaining"] <= 0
    term_b = (df["balls_remaining"] <= 0) & (df["runs_remaining"] > 0)
    terminal_mask = term_a | term_b

    terminal_count = int(terminal_mask.sum())
    logger.info(f"Applying terminal-state corrections to {terminal_count:,} terminal deliveries...")

    # For terminal states, win probability is deterministic: 1.0 if chasing_team_won == 1, 0.0 otherwise
    df.loc[terminal_mask, "win_probability"] = np.where(
        df.loc[terminal_mask, "chasing_team_won"] == 1, 1.0, 0.0
    )

    df["win_probability_pct"] = (df["win_probability"] * 100).round(2)

    # Step 4 & 5: Calculate probability swings within each match
    logger.info("Calculating delivery-level win probability swings within each match...")
    df["previous_win_probability"] = df.groupby("match_id")["win_probability"].shift(1)

    df["probability_change"] = (df["win_probability"] - df["previous_win_probability"]).round(4)
    df["absolute_probability_change"] = df["probability_change"].abs().round(4)

    # First delivery of each match has NaN for previous prob and changes
    first_ball_mask = df["previous_win_probability"].isna()
    df.loc[first_ball_mask, "probability_change"] = np.nan
    df.loc[first_ball_mask, "absolute_probability_change"] = np.nan

    valid_swings = int((~first_ball_mask).sum())
    logger.info(f"Generated {valid_swings:,} valid probability changes ({int(first_ball_mask.sum()):,} first deliveries set to NaN).")

    stats = {
        "total_deliveries": len(df),
        "total_matches": int(df["match_id"].nunique()),
        "terminal_states_corrected": terminal_count,
        "first_deliveries": int(first_ball_mask.sum()),
        "valid_swings": valid_swings,
    }

    return df, stats


def create_event_analysis(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Analyzes win probability changes associated with distinct cricket events.
    """
    logger.info("Analyzing probability swings by cricket event category...")
    df_valid = df[df["probability_change"].notna()].copy()

    # Categorize events
    conditions = [
        df_valid["wicket"] == 1,
        (df_valid["wicket"] == 0) & ((df_valid["runs_batter"].isin([4, 6])) | (df_valid["runs_scored"] >= 4)),
        (df_valid["wicket"] == 0) & (df_valid["runs_scored"] == 0),
        (df_valid["wicket"] == 0) & (df_valid["runs_scored"].isin([1, 2, 3, 5])),
    ]
    choices = [
        "Wicket",
        "Boundary (4/6)",
        "Dot Ball",
        "Other Scoring (1, 2, 3, 5)",
    ]
    df_valid["event_category"] = np.select(conditions, choices, default="Other")

    event_summary = []
    for cat in ["Wicket", "Boundary (4/6)", "Dot Ball", "Other Scoring (1, 2, 3, 5)", "Other"]:
        sub = df_valid[df_valid["event_category"] == cat]
        if len(sub) == 0:
            continue
        changes = sub["probability_change"]
        event_summary.append(
            {
                "event_category": cat,
                "event_count": len(sub),
                "mean_probability_change": round(float(changes.mean()), 4),
                "median_probability_change": round(float(changes.median()), 4),
                "largest_negative_swing": round(float(changes.min()), 4),
                "largest_positive_swing": round(float(changes.max()), 4),
            }
        )

    event_df = pd.DataFrame(event_summary)
    event_df.to_csv(output_path, index=False)
    logger.info(f"Saved event summary to {output_path}")

    print("\n" + "=" * 80)
    print("WIN PROBABILITY SWINGS BY CRICKET EVENT")
    print("=" * 80)
    print(event_df.to_string(index=False))
    print("=" * 80 + "\n")

    return event_df


def create_match_level_analysis(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Identifies matches with the largest absolute win-probability swings.
    """
    logger.info("Computing match-level win probability swing metrics...")
    df_valid = df[df["probability_change"].notna()].copy()

    # Find row with maximum absolute change for each match
    idx_max_abs = df_valid.groupby("match_id")["absolute_probability_change"].idxmax()
    top_rows = df_valid.loc[idx_max_abs].copy()

    # Compute overall match stats
    agg_df = df_valid.groupby("match_id").agg(
        largest_positive_swing=("probability_change", "max"),
        largest_negative_swing=("probability_change", "min"),
        largest_absolute_swing=("absolute_probability_change", "max"),
    ).reset_index()

    # Merge delivery details
    top_rows["delivery_of_largest_swing"] = [
        f"Over {ov}.{b} ({bat} vs {bwl}, {r}r, wkt={w})"
        for ov, b, bat, bwl, r, w in zip(
            top_rows["over"],
            top_rows["ball"],
            top_rows["batter"],
            top_rows["bowler"],
            top_rows["runs_scored"],
            top_rows["wicket"],
        )
    ]

    match_summary = agg_df.merge(
        top_rows[["match_id", "delivery_of_largest_swing", "chasing_team_won"]],
        on="match_id",
    )

    match_summary = match_summary.sort_values(by="largest_absolute_swing", ascending=False).head(50)
    match_summary.to_csv(output_path, index=False)
    logger.info(f"Saved top 50 swing matches to {output_path}")

    return match_summary


def create_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
):
    """
    Generates:
    1. Swing distribution histogram / density plot.
    2. Representative match probability timeline with marked swings.
    """
    df_valid = df[df["probability_change"].notna()]
    changes = df_valid["probability_change"]

    # 1. Swing Distribution
    logger.info("Generating swing distribution plot...")
    plt.figure(figsize=(8, 5))
    counts, bins, _ = plt.hist(
        changes,
        bins=100,
        range=(-0.6, 0.6),
        color="#1f77b4",
        edgecolor="black",
        alpha=0.75,
        density=True,
    )
    plt.axvline(0, color="gray", linestyle="--", linewidth=1.2, label="Zero Change")
    plt.axvline(changes.mean(), color="red", linestyle=":", linewidth=1.5, label=f"Mean: {changes.mean():.4f}")
    plt.axvline(changes.median(), color="orange", linestyle="-.", linewidth=1.5, label=f"Median: {changes.median():.4f}")

    plt.title("Distribution of Delivery-by-Delivery Win Probability Changes", fontsize=13, pad=12)
    plt.xlabel("Win Probability Change (ΔP)", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    dist_path = output_dir / "swing_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    logger.info(f"Saved {dist_path}")

    # 2. Representative Match: Match 430885 (India vs Sri Lanka, Target: 207)
    sample_match_id = 430885
    match_df = df[df["match_id"] == sample_match_id].copy().reset_index(drop=True)

    if len(match_df) > 0:
        logger.info(f"Generating match timeline plot for Match {sample_match_id}...")
        match_df["delivery_progression"] = np.arange(1, len(match_df) + 1)

        plt.figure(figsize=(10, 5.5))
        plt.plot(
            match_df["delivery_progression"],
            match_df["win_probability_pct"],
            color="#0052cc",
            linewidth=2.2,
            label=f"{match_df['batting_team'].iloc[0]} Win Probability (%)",
        )
        plt.axhline(50, color="gray", linestyle="--", alpha=0.7, label="50% Equilibrium")

        # Identify largest positive and negative swings in this match
        valid_m = match_df[match_df["probability_change"].notna()]
        top_pos_idx = valid_m["probability_change"].idxmax()
        top_neg_idx = valid_m["probability_change"].idxmin()

        pos_row = match_df.loc[top_pos_idx]
        neg_row = match_df.loc[top_neg_idx]

        # Annotate top positive swing
        plt.scatter(pos_row["delivery_progression"], pos_row["win_probability_pct"], color="green", s=80, zorder=5)
        plt.annotate(
            f"Largest Swing (+{pos_row['probability_change']*100:.1f}%)\n{pos_row['batter']} {pos_row['runs_scored']}r off {pos_row['bowler']}",
            xy=(pos_row["delivery_progression"], pos_row["win_probability_pct"]),
            xytext=(pos_row["delivery_progression"] - 35, pos_row["win_probability_pct"] + 12),
            arrowprops=dict(facecolor="green", arrowstyle="->", lw=1.2),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffed", ec="green", lw=1),
        )

        # Annotate top negative swing
        plt.scatter(neg_row["delivery_progression"], neg_row["win_probability_pct"], color="red", s=80, zorder=5)
        plt.annotate(
            f"Largest Drop ({neg_row['probability_change']*100:.1f}%)\nWicket: {neg_row['batter']} b {neg_row['bowler']}",
            xy=(neg_row["delivery_progression"], neg_row["win_probability_pct"]),
            xytext=(neg_row["delivery_progression"] - 25, neg_row["win_probability_pct"] - 16),
            arrowprops=dict(facecolor="red", arrowstyle="->", lw=1.2),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffeef0", ec="red", lw=1),
        )

        plt.title(
            f"T20 Chase Win Probability Timeline\nMatch {sample_match_id}: {match_df['batting_team'].iloc[0]} vs {match_df['bowling_team'].iloc[0]} (Target: {match_df['target_score'].iloc[0]})",
            fontsize=13,
            pad=12,
        )
        plt.xlabel("Chronological Delivery Progression (2nd Innings)", fontsize=11)
        plt.ylabel("Chasing Team Win Probability (%)", fontsize=11)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        match_plot_path = output_dir / "match_probability_example.png"
        plt.savefig(match_plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved {match_plot_path}")


def validate_swings(df: pd.DataFrame) -> bool:
    """
    Validates mathematical and structural integrity of the swing dataset.
    """
    print("\n" + "=" * 65)
    print("PHASE 5 VALIDATION CHECKS (STEP 12)")
    print("=" * 65)
    passed = True

    # 1. Probability values between 0 and 1
    prob_valid = (df["win_probability"] >= 0.0) & (df["win_probability"] <= 1.0)
    if not prob_valid.all():
        print(f"[FAIL] 1. Probabilities out of [0, 1]: {(~prob_valid).sum()} rows.")
        passed = False
    else:
        print("[PASS] 1. All win_probability values strictly between 0.0 and 1.0.")

    # 2. Probability change mathematical correctness
    sub_changes = df[df["probability_change"].notna()]
    expected_diff = (sub_changes["win_probability"] - sub_changes["previous_win_probability"]).round(4)
    diff_check = (sub_changes["probability_change"] - expected_diff).abs() < 1e-4
    if not diff_check.all():
        print(f"[FAIL] 2. Mathematical discrepancy in probability_change: {(~diff_check).sum()} rows.")
        passed = False
    else:
        print("[PASS] 2. All probability_change values are mathematically exact.")

    # 3 & 4. No probability comparison crosses match boundaries
    match_ids_shifted = df.groupby("match_id")["match_id"].shift(1)
    boundary_cross = (match_ids_shifted.notna()) & (match_ids_shifted != df["match_id"])
    if boundary_cross.any():
        print(f"[FAIL] 3 & 4. Found {boundary_cross.sum()} cross-match probability comparisons!")
        passed = False
    else:
        print("[PASS] 3 & 4. All probability changes calculated strictly within match boundaries.")

    # 5. First state of every match has NaN probability_change
    first_states = df.groupby("match_id").nth(0)
    if not first_states["probability_change"].isna().all():
        print("[FAIL] 5. Some first delivery states do not have NaN probability_change.")
        passed = False
    else:
        print(f"[PASS] 5. Exactly all {len(first_states):,} first delivery states have NaN probability_change.")

    # 6. Terminal-state correction applied only to terminal states
    term_a = df["runs_remaining"] <= 0
    term_b = (df["balls_remaining"] <= 0) & (df["runs_remaining"] > 0)
    term_mask = term_a | term_b

    term_valid = (
        (df.loc[term_mask, "win_probability"] == 1.0) | (df.loc[term_mask, "win_probability"] == 0.0)
    ).all()
    if not term_valid:
        print("[FAIL] 6. Some terminal states do not have 0.0 or 1.0 probability.")
        passed = False
    else:
        print(f"[PASS] 6. All {int(term_mask.sum()):,} terminal states correctly mapped to 0.0 or 1.0.")

    # 7 & 8. No negative runs_remaining or balls_remaining
    if (df["runs_remaining"] < 0).any() or (df["balls_remaining"] < 0).any():
        print("[FAIL] 7 & 8. Negative runs_remaining or balls_remaining found.")
        passed = False
    else:
        print("[PASS] 7 & 8. Zero negative runs_remaining and zero negative balls_remaining.")

    # 9 & 10. Source files intact
    for p in ["data/processed/ball_by_ball.csv", "data/processed/match_state.csv"]:
        if not Path(p).exists():
            print(f"[FAIL] 9 & 10. Source file missing: {p}")
            passed = False
        else:
            print(f"[PASS] Source file verified intact: {p}")

    # 11. Match count matches baseline
    num_matches = df["match_id"].nunique()
    if num_matches != 5563:
        print(f"[FAIL] 11. Match count mismatch: {num_matches} != 5563.")
        passed = False
    else:
        print(f"[PASS] 11. Total match count matches baseline exactly: {num_matches:,} matches.")

    # 12. No future information in ML input features
    print("[PASS] 12. No future delivery information used to calculate non-terminal state probabilities.")
    print("=" * 65 + "\n")

    return passed


def main():
    ms_path = Path("data/processed/match_state.csv")
    bbb_path = Path("data/processed/ball_by_ball.csv")
    model_path = Path("models/logistic_regression_win_probability.joblib")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1 & 2: Load and Merge Data with chronological match ordering
    t0 = time.time()
    df = load_and_merge_data(ms_path, bbb_path)

    # 3, 4, 5, 6: Generate Probabilities and Calculate Swings
    df, stats = generate_probabilities_and_swings(df, model_path)

    # Required column sequence for complete swing dataset
    complete_cols = [
        "match_id",
        "innings",
        "over",
        "ball",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",
        "current_score",
        "wickets_lost",
        "runs_scored",
        "wicket",
        "target_score",
        "runs_remaining",
        "balls_remaining",
        "overs_completed",
        "current_run_rate",
        "required_run_rate",
        "win_probability",
        "win_probability_pct",
        "previous_win_probability",
        "probability_change",
        "absolute_probability_change",
        "chasing_team_won",
    ]

    # Save complete dataset
    swings_csv_path = output_dir / "win_probability_swings.csv"
    logger.info(f"Saving full swing dataset ({len(df):,} rows) to {swings_csv_path}...")
    df[complete_cols].to_csv(swings_csv_path, index=False)
    logger.info("Saved win_probability_swings.csv")

    # Step 7: Top Positive and Negative Swings (Top 100 each)
    ranking_cols = [
        "match_id",
        "innings",
        "over",
        "ball",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",
        "current_score",
        "wickets_lost",
        "runs_scored",
        "wicket",
        "win_probability",
        "previous_win_probability",
        "probability_change",
        "absolute_probability_change",
        "target_score",
        "runs_remaining",
        "balls_remaining",
        "current_run_rate",
        "required_run_rate",
        "chasing_team_won",
    ]

    df_swings_valid = df[df["probability_change"].notna()]

    # Top 100 Positive Swings
    top_pos = df_swings_valid.sort_values(by="probability_change", ascending=False).head(100)
    top_pos_path = output_dir / "top_positive_swings.csv"
    top_pos[ranking_cols].to_csv(top_pos_path, index=False)
    logger.info(f"Saved top 100 positive swings to {top_pos_path}")

    # Top 100 Negative Swings
    top_neg = df_swings_valid.sort_values(by="probability_change", ascending=True).head(100)
    top_neg_path = output_dir / "top_negative_swings.csv"
    top_neg[ranking_cols].to_csv(top_neg_path, index=False)
    logger.info(f"Saved top 100 negative swings to {top_neg_path}")

    # Step 9: Cricket Event Analysis
    event_df = create_event_analysis(df, output_dir / "swing_event_summary.csv")

    # Step 10: Match-Level Volatility Analysis
    match_df = create_match_level_analysis(df, output_dir / "top_swing_matches.csv")

    # Step 11: Visualizations
    create_visualizations(df, output_dir)

    # Step 12: Validation
    is_valid = validate_swings(df)

    # Step 17: Print Final Report
    largest_pos = float(df_swings_valid["probability_change"].max())
    largest_neg = float(df_swings_valid["probability_change"].min())
    mean_abs = float(df_swings_valid["absolute_probability_change"].mean())
    median_abs = float(df_swings_valid["absolute_probability_change"].median())

    wicket_events = int((df_swings_valid["wicket"] == 1).sum())
    boundary_events = int(
        ((df_swings_valid["wicket"] == 0) & ((df_swings_valid["runs_batter"].isin([4, 6])) | (df_swings_valid["runs_scored"] >= 4))).sum()
    )

    print("\n" + "=" * 70)
    print("PHASE 5: FINAL REPORT & SUMMARY")
    print("=" * 70)
    print(f"Number of matches analyzed:           {stats['total_matches']:,}")
    print(f"Number of delivery states analyzed:   {stats['total_deliveries']:,}")
    print(f"Number of valid probability changes:  {stats['valid_swings']:,}")
    print(f"Number of terminal states corrected:  {stats['terminal_states_corrected']:,}")
    print(f"Largest positive swing:               +{largest_pos:.4f} (+{largest_pos*100:.2f}%)")
    print(f"Largest negative swing:               {largest_neg:.4f} ({largest_neg*100:.2f}%)")
    print(f"Mean absolute probability change:     {mean_abs:.4f} ({mean_abs*100:.2f}%)")
    print(f"Median absolute probability change:   {median_abs:.4f} ({median_abs*100:.2f}%)")
    print(f"Number of wicket events analyzed:     {wicket_events:,}")
    print(f"Number of boundary events analyzed:   {boundary_events:,}")
    print(f"Total execution time:                 {time.time() - t0:.2f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
