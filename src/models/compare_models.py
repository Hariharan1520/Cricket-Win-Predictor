"""
Phase 4: Model Comparison Pipeline (Logistic Regression vs Random Forest)
Evaluates and compares the baseline Logistic Regression model against
a Random Forest classifier on the exact same match-level test partition.

Outputs:
    - models/random_forest_win_probability.joblib
    - outputs/model_comparison.json
    - outputs/model_comparison_roc_curve.png
    - outputs/model_comparison_calibration_curve.png
    - outputs/logistic_confusion_matrix.png
    - outputs/random_forest_confusion_matrix.png
    - outputs/random_forest_feature_importance.csv
    - outputs/random_forest_feature_importance.png
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

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
TARGET_COL = "chasing_team_won"
RANDOM_STATE = 42


def load_and_split_data(
    data_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Reproduces the exact same match-level train/test split from Phase 3.
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)

    unique_matches = df["match_id"].unique()
    train_matches, test_matches = train_test_split(
        unique_matches,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_set = set(train_matches)
    test_set = set(test_matches)
    assert len(train_set.intersection(test_set)) == 0, "Match overlap detected!"

    train_df = df[df["match_id"].isin(train_set)].copy()
    test_df = df[df["match_id"].isin(test_set)].copy()

    logger.info(f"Training matches: {len(train_matches):,} | Rows: {len(train_df):,}")
    logger.info(f"Testing matches:  {len(test_matches):,} | Rows: {len(test_df):,}")

    return train_df, test_df, list(train_matches), list(test_matches)


def plot_single_confusion_matrix(
    cm: np.ndarray,
    title: str,
    output_path: Path,
    cmap=plt.cm.Blues,
):
    """Plots and saves a formatted confusion matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=13, pad=15)
    plt.colorbar()
    classes = ["Defending Won (0)", "Chasing Won (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=15)
    plt.yticks(tick_marks, classes)

    total = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total) * 100
            plt.text(
                j,
                i,
                f"{val:,}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=11,
            )

    plt.ylabel("Actual Outcome", fontsize=11)
    plt.xlabel("Predicted Outcome", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_model_comparison():
    data_path = Path("data/processed/match_state.csv")
    model_dir = Path("models")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reproduce exact match-level split
    train_df, test_df, train_matches, test_matches = load_and_split_data(data_path)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].values

    # 2. Load existing Logistic Regression pipeline
    lr_model_path = model_dir / "logistic_regression_win_probability.joblib"
    logger.info(f"Loading Logistic Regression baseline from {lr_model_path}...")
    lr_pipeline = joblib.load(lr_model_path)

    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    # 3. Train Random Forest (no scaling, unnormalized raw numerical features)
    logger.info("Training Random Forest Classifier (n_estimators=300, max_depth=16, min_samples_leaf=5)...")
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train)
    logger.info("Random Forest training complete.")

    # Save Random Forest model
    rf_model_path = model_dir / "random_forest_win_probability.joblib"
    joblib.dump(rf_clf, rf_model_path)
    logger.info(f"Saved Random Forest model to {rf_model_path}")

    # Generate predictions
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

    # 4. Metric Calculations
    lr_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_lr)), 4),
        "precision": round(float(precision_score(y_test, y_pred_lr)), 4),
        "recall": round(float(recall_score(y_test, y_pred_lr)), 4),
        "f1": round(float(f1_score(y_test, y_pred_lr)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_lr)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob_lr)), 4),
    }

    rf_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_rf)), 4),
        "precision": round(float(precision_score(y_test, y_pred_rf)), 4),
        "recall": round(float(recall_score(y_test, y_pred_rf)), 4),
        "f1": round(float(f1_score(y_test, y_pred_rf)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_rf)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob_rf)), 4),
    }

    # Save model_comparison.json
    comparison_json = {
        "random_state": RANDOM_STATE,
        "feature_names": FEATURE_COLS,
        "train_match_count": len(train_matches),
        "test_match_count": len(test_matches),
        "train_row_count": len(train_df),
        "test_row_count": len(test_df),
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
    }
    with open(output_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_json, f, indent=4)
    logger.info("Saved model_comparison.json")

    # 5. Combined ROC Curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_lr, tpr_lr, color="#1f77b4", linewidth=2, label=f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.4f})")
    plt.plot(fpr_rf, tpr_rf, color="#2ca02c", linewidth=2, linestyle="-.", label=f"Random Forest (AUC = {rf_metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.title("Model Comparison: ROC Curves", fontsize=14, pad=15)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_comp_path = output_dir / "model_comparison_roc_curve.png"
    plt.savefig(roc_comp_path, dpi=300)
    plt.close()
    logger.info(f"Saved {roc_comp_path}")

    # 6. Combined Calibration Curve
    prob_true_lr, prob_pred_lr = calibration_curve(y_test, y_prob_lr, n_bins=10, strategy="uniform")
    prob_true_rf, prob_pred_rf = calibration_curve(y_test, y_prob_rf, n_bins=10, strategy="uniform")

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred_lr, prob_true_lr, marker="o", linewidth=2, color="#1f77b4", label=f"Logistic Regression (Brier = {lr_metrics['brier_score']:.4f})")
    plt.plot(prob_pred_rf, prob_true_rf, marker="s", linewidth=2, color="#2ca02c", linestyle="-.", label=f"Random Forest (Brier = {rf_metrics['brier_score']:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect Calibration")
    plt.title("Model Comparison: Probability Calibration (Reliability)", fontsize=14, pad=15)
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Observed Fraction of Positives", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    cal_comp_path = output_dir / "model_comparison_calibration_curve.png"
    plt.savefig(cal_comp_path, dpi=300)
    plt.close()
    logger.info(f"Saved {cal_comp_path}")

    # 7. Confusion Matrices
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plot_single_confusion_matrix(
        cm_lr,
        "Logistic Regression Confusion Matrix (Threshold = 0.50)",
        output_dir / "logistic_confusion_matrix.png",
        cmap=plt.cm.Blues,
    )
    plot_single_confusion_matrix(
        cm_rf,
        "Random Forest Confusion Matrix (Threshold = 0.50)",
        output_dir / "random_forest_confusion_matrix.png",
        cmap=plt.cm.Greens,
    )
    logger.info("Saved confusion matrices.")

    # 8. Random Forest Feature Importance
    importances = rf_clf.feature_importances_
    fi_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": np.round(importances, 4),
        }
    ).sort_values(by="importance", ascending=False)
    fi_csv_path = output_dir / "random_forest_feature_importance.csv"
    fi_df.to_csv(fi_csv_path, index=False)
    logger.info(f"Saved {fi_csv_path}")

    # Plot Feature Importance
    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(fi_df))
    plt.barh(y_pos, fi_df["importance"].values[::-1], color="#2ca02c", alpha=0.85)
    plt.yticks(y_pos, fi_df["feature"].values[::-1], fontsize=11)
    plt.xlabel("Gini Feature Importance (Model-Based)", fontsize=12)
    plt.title("Random Forest Feature Importance", fontsize=14, pad=15)
    plt.tight_layout()
    fi_png_path = output_dir / "random_forest_feature_importance.png"
    plt.savefig(fi_png_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fi_png_path}")

    # 9. Print Comparison Table
    print("\n" + "=" * 65)
    print("PHASE 4: MODEL COMPARISON TABLE (HELD-OUT TEST SET)")
    print("=" * 65)
    print(f"{'Metric':<18} | {'Logistic Regression':<20} | {'Random Forest':<15}")
    print("-" * 65)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]:
        lr_val = lr_metrics[metric]
        rf_val = rf_metrics[metric]
        print(f"{metric.upper():<18} | {lr_val:<20.4f} | {rf_val:<15.4f}")
    print("=" * 65)

    print("\nRandom Forest Feature Importances (Model-based Gini reduction):")
    print(fi_df.to_string(index=False))

    # 10. Model Agreement & Disagreement Analysis
    test_df_analysis = test_df.copy()
    test_df_analysis["lr_prob"] = np.round(y_prob_lr * 100, 1)
    test_df_analysis["rf_prob"] = np.round(y_prob_rf * 100, 1)
    test_df_analysis["abs_prob_diff"] = np.round(np.abs(y_prob_lr - y_prob_rf) * 100, 1)

    mean_diff = test_df_analysis["abs_prob_diff"].mean()
    median_diff = test_df_analysis["abs_prob_diff"].median()
    max_diff = test_df_analysis["abs_prob_diff"].max()
    agree_within_5 = (test_df_analysis["abs_prob_diff"] <= 5.0).mean() * 100
    agree_within_10 = (test_df_analysis["abs_prob_diff"] <= 10.0).mean() * 100

    print("\n" + "=" * 65)
    print("MODEL AGREEMENT ANALYSIS")
    print("=" * 65)
    print(f"Mean Absolute Probability Difference:   {mean_diff:.2f}%")
    print(f"Median Absolute Probability Difference: {median_diff:.2f}%")
    print(f"Maximum Probability Difference:        {max_diff:.2f}%")
    print(f"Predictions agreeing within 5.0%:       {agree_within_5:.1f}%")
    print(f"Predictions agreeing within 10.0%:      {agree_within_10:.1f}%")

    print("\nSample High-Disagreement States (|LR - RF| > 30%):")
    high_disagree = test_df_analysis.sort_values(by="abs_prob_diff", ascending=False).head(5)
    disp_cols = [
        "match_id",
        "current_score",
        "wickets_lost",
        "runs_remaining",
        "balls_remaining",
        "current_run_rate",
        "required_run_rate",
        "chasing_team_won",
        "lr_prob",
        "rf_prob",
        "abs_prob_diff",
    ]
    print(high_disagree[disp_cols].to_string(index=False))

    # 11. Diverse Test Set Predictions Across Scenarios
    print("\n" + "=" * 90)
    print("SAMPLE TEST PREDICTIONS ACROSS DIVERSE MATCH CONTEXTS")
    print("=" * 90)
    early = test_df_analysis[test_df_analysis["overs_completed"] <= 3.0].sample(2, random_state=42)
    middle = test_df_analysis[(test_df_analysis["overs_completed"] >= 8.0) & (test_df_analysis["overs_completed"] <= 14.0)].sample(3, random_state=42)
    death_easy = test_df_analysis[(test_df_analysis["overs_completed"] >= 16.0) & (test_df_analysis["required_run_rate"] <= 6.0) & (test_df_analysis["wickets_lost"] <= 5)].sample(2, random_state=42)
    death_hard = test_df_analysis[(test_df_analysis["overs_completed"] >= 16.0) & (test_df_analysis["required_run_rate"] >= 12.0) & (test_df_analysis["wickets_lost"] >= 6)].sample(2, random_state=42)
    tight = test_df_analysis[(test_df_analysis["balls_remaining"].between(6, 18)) & (test_df_analysis["required_run_rate"].between(8.0, 11.0))].sample(2, random_state=42)

    sample_comp = pd.concat([early, middle, death_easy, death_hard, tight]).drop_duplicates().head(11)
    print(sample_comp[disp_cols].to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_model_comparison()
