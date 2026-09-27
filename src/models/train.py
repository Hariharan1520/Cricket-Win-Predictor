"""
Phase 3: Baseline Win-Probability Model Training Pipeline
Trains a Logistic Regression baseline model on match-state delivery data
using match-level train/test splitting and standardized pipeline evaluation.

Outputs:
    - models/logistic_regression_win_probability.joblib
    - outputs/model_metrics.json
    - outputs/logistic_coefficients.csv
    - outputs/confusion_matrix.png
    - outputs/roc_curve.png
    - outputs/calibration_curve.png
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
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Core modelling features
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
    Loads match_state.csv and performs a strict match-level train/test split.
    """
    logger.info(f"Loading match state data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)

    unique_matches = df["match_id"].unique()
    total_matches = len(unique_matches)
    logger.info(f"Total unique matches: {total_matches:,}")

    train_matches, test_matches = train_test_split(
        unique_matches,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_set = set(train_matches)
    test_set = set(test_matches)
    assert len(train_set.intersection(test_set)) == 0, "Data leakage: match overlap detected!"

    train_df = df[df["match_id"].isin(train_set)].copy()
    test_df = df[df["match_id"].isin(test_set)].copy()

    logger.info(f"Training matches: {len(train_matches):,} | Rows: {len(train_df):,}")
    logger.info(f"Testing matches:  {len(test_matches):,} | Rows: {len(test_df):,}")

    return train_df, test_df, list(train_matches), list(test_matches)


def train_baseline_model(train_df: pd.DataFrame) -> Pipeline:
    """
    Constructs and fits the StandardScaler + LogisticRegression pipeline.
    """
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    logger.info("Fitting Logistic Regression pipeline on training matches...")
    pipeline.fit(X_train, y_train)
    logger.info("Model fitting complete.")

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    train_matches: List[int],
    test_matches: List[int],
    output_dir: Path,
) -> Dict[str, float]:
    """
    Evaluates the model on held-out test data and saves plots, metrics, and coefficients.
    """
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].values

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))

    metrics = {
        "model_name": "Logistic Regression (StandardScaler + LogisticRegression)",
        "random_state": RANDOM_STATE,
        "features": FEATURE_COLS,
        "total_matches": len(train_matches) + len(test_matches),
        "training_matches": len(train_matches),
        "testing_matches": len(test_matches),
        "training_rows": len(train_df),
        "testing_rows": len(test_df),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
    }

    # Save metrics JSON
    metrics_path = output_dir / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to {metrics_path}")

    # Step 7: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Test Set Confusion Matrix", fontsize=14, pad=15)
    plt.colorbar()
    classes = ["Defending Won (0)", "Chasing Won (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=15)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / len(y_test)) * 100
            label = f"{val:,}\n({pct:.1f}%)"
            plt.text(
                j,
                i,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=11,
            )

    plt.ylabel("Actual Outcome", fontsize=12)
    plt.xlabel("Predicted Outcome", fontsize=12)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix to {cm_path}")

    # Step 8: Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, color="#1f77b4", label=f"Logistic Regression (Brier: {brier:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.title("Probability Calibration Curve (Reliability Diagram)", fontsize=14, pad=15)
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Actual Win Rate)", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    cal_path = output_dir / "calibration_curve.png"
    plt.savefig(cal_path, dpi=300)
    plt.close()
    logger.info(f"Saved calibration curve to {cal_path}")

    # Step 9: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#ff7f0e", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier (AUC = 0.50)")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, pad=15)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve to {roc_path}")

    # Step 14: Model Interpretability (Coefficients)
    clf = pipeline.named_steps["classifier"]
    coefs = clf.coef_[0]
    intercept = clf.intercept_[0]

    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coefficient": np.round(coefs, 4),
            "effect_direction": [
                "Positive (higher win prob)" if c > 0 else "Negative (lower win prob)"
                for c in coefs
            ],
        }
    ).sort_values(by="coefficient", ascending=False)

    coef_path = output_dir / "logistic_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    logger.info(f"Saved coefficients to {coef_path}")

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE ON HELD-OUT TEST MATCHES")
    print("=" * 65)
    print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f} (Lower = superior probability calibration)")
    print(f"Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
    print("\nLogistic Regression Coefficients (Standardized Scale):")
    print(f"Intercept: {intercept:.4f}")
    print(coef_df.to_string(index=False))
    print("=" * 65 + "\n")

    return metrics


def generate_sample_predictions(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selects diverse delivery states from test matches and shows predictions.
    """
    X_test = test_df[FEATURE_COLS]
    test_df_copy = test_df.copy()
    test_df_copy["predicted_chase_win_prob"] = np.round(pipeline.predict_proba(X_test)[:, 1] * 100, 1)

    # Pick samples representing diverse match situations
    # 1. Early innings (overs_completed < 4)
    early = test_df_copy[test_df_copy["overs_completed"] <= 3.0].sample(2, random_state=42)
    # 2. Middle overs (8 < overs < 14)
    middle = test_df_copy[(test_df_copy["overs_completed"] >= 8.0) & (test_df_copy["overs_completed"] <= 14.0)].sample(3, random_state=42)
    # 3. Death overs (overs > 17) with low RRR (comfortable chase)
    death_easy = test_df_copy[(test_df_copy["overs_completed"] >= 16.0) & (test_df_copy["required_run_rate"] <= 6.0) & (test_df_copy["wickets_lost"] <= 5)].sample(2, random_state=42)
    # 4. Death overs with high RRR (difficult chase)
    death_hard = test_df_copy[(test_df_copy["overs_completed"] >= 16.0) & (test_df_copy["required_run_rate"] >= 12.0) & (test_df_copy["wickets_lost"] >= 6)].sample(2, random_state=42)
    # 5. Tight brink situations (balls remaining between 6 and 18, RRR around 8-10)
    tight = test_df_copy[(test_df_copy["balls_remaining"].between(6, 18)) & (test_df_copy["required_run_rate"].between(8.0, 11.0))].sample(2, random_state=42)

    samples = pd.concat([early, middle, death_easy, death_hard, tight]).drop_duplicates().head(11)

    display_cols = [
        "match_id",
        "current_score",
        "wickets_lost",
        "runs_remaining",
        "balls_remaining",
        "current_run_rate",
        "required_run_rate",
        "chasing_team_won",
        "predicted_chase_win_prob",
    ]
    sample_view = samples[display_cols].rename(
        columns={
            "chasing_team_won": "actual_outcome",
            "predicted_chase_win_prob": "win_prob_%",
        }
    )

    print("=" * 80)
    print("SAMPLE TEST PREDICTIONS ACROSS DIVERSE MATCH CONTEXTS")
    print("=" * 80)
    print(sample_view.to_string(index=False))
    print("=" * 80 + "\n")
    return sample_view


def main():
    data_path = Path("data/processed/match_state.csv")
    model_dir = Path("models")
    output_dir = Path("outputs")

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1, 2, 3: Load & Match-Level Split
    train_df, test_df, train_matches, test_matches = load_and_split_data(data_path)

    # Step 4 & 5: Build Pipeline & Train
    pipeline = train_baseline_model(train_df)

    # Step 11: Save Pipeline Model
    model_path = model_dir / "logistic_regression_win_probability.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved trained pipeline to {model_path}")

    # Step 6, 7, 8, 9, 12, 14: Evaluate, Plot, Save Metrics & Coefficients
    evaluate_model(
        pipeline=pipeline,
        test_df=test_df,
        train_df=train_df,
        train_matches=train_matches,
        test_matches=test_matches,
        output_dir=output_dir,
    )

    # Step 10: Generate and Print Sample Predictions
    generate_sample_predictions(pipeline, test_df)


if __name__ == "__main__":
    main()
