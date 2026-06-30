"""
train_model.py
==============
Full training pipeline for Employee Attrition Prediction.

Steps executed:
  1. Load dataset
  2. Preprocessing pipeline (clean → encode → scale)
  3. Train Logistic Regression, Random Forest, XGBoost
  4. Evaluate all models and print comparison table
  5. Save the best model (ROC-AUC criterion)
  6. Generate SHAP explainability visualizations

Run:
    python train_model.py

Author  : Stellar Intelligence
Version : 1.0.0
"""

import logging
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
)

from employee_attrition import (
    DATASET_PATH,
    MODEL_PATH,
    DataPreprocessor,
    ModelTrainer,
    load_dataset,
)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# SHAP Explainability
# ===========================================================================

def generate_shap_plots(
    model,
    model_name: str,
    X_test: np.ndarray,
    feature_names: list,
) -> None:
    """
    Generate SHAP visualisations for the best model.

    SHAP (SHapley Additive exPlanations) quantifies how much each
    feature pushes a prediction away from the baseline (expected value).

    Plots saved:
    - shap_summary.png         → beeswarm showing feature impact direction
    - shap_feature_importance.png → bar chart of mean |SHAP| values
    - shap_waterfall.png       → single-sample waterfall (first test row)
    """
    logger.info("─" * 55)
    logger.info("SHAP EXPLAINABILITY")
    logger.info("─" * 55)

    # ---- Build SHAP explainer based on model type --------------------
    if "XGBoost" in model_name:
        # TreeExplainer is fast and exact for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # For binary classification shap_values may be a list; take class 1
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
    elif "Random Forest" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
    else:
        # KernelExplainer works for any model but is slower
        # Use a small background sample for speed
        background = shap.sample(X_test, 50, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values_full = explainer.shap_values(X_test[:100], nsamples=100)
        if isinstance(shap_values_full, list):
            sv = shap_values_full[1]
        else:
            sv = shap_values_full
        X_test = X_test[:100]  # trim to match shap values length

    # ---- 1. Summary Plot (beeswarm) ----------------------------------
    # Each dot = one test sample; x-axis = SHAP value (impact on output)
    # Colour = feature value (red = high, blue = low)
    logger.info("  Generating SHAP summary plot …")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, X_test,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary Plot — {model_name}", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("shap_summary.png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close()
    logger.info("  Saved → shap_summary.png")

    # ---- 2. Feature Importance (bar) ---------------------------------
    # Mean absolute SHAP value per feature — higher = more important
    logger.info("  Generating SHAP feature importance …")
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        sv, X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Feature Importance — {model_name}", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close()
    logger.info("  Saved → shap_feature_importance.png")

    # ---- 3. Waterfall Plot (single sample) ---------------------------
    # Shows how each feature pushes the prediction for the first test row
    logger.info("  Generating SHAP waterfall plot …")
    try:
        explanation = shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value if not isinstance(
                explainer.expected_value, list
            ) else explainer.expected_value[1],
            data=X_test[0],
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(explanation, show=False, max_display=15)
        plt.title(f"SHAP Waterfall — {model_name} (Sample #1)", fontweight="bold")
        plt.tight_layout()
        plt.savefig("shap_waterfall.png", bbox_inches="tight", facecolor="white", dpi=120)
        plt.close()
        logger.info("  Saved → shap_waterfall.png")
    except Exception as exc:
        logger.warning(f"  Waterfall plot skipped: {exc}")


# ===========================================================================
# Evaluation plots
# ===========================================================================

def generate_evaluation_plots(
    trainer: "ModelTrainer",
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Save Confusion Matrix, ROC Curve, and Precision-Recall Curve
    for the best model.
    """
    logger.info("─" * 55)
    logger.info("EVALUATION PLOTS")
    logger.info("─" * 55)

    model = trainer.best_model
    name = trainer.best_model_name

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stay", "Leave"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig("eval_confusion_matrix.png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close()
    logger.info("  Saved → eval_confusion_matrix.png")

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("eval_roc_curve.png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close()
    logger.info("  Saved → eval_roc_curve.png")

    # 3. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    plt.tight_layout()
    plt.savefig("eval_precision_recall.png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close()
    logger.info("  Saved → eval_precision_recall.png")


# ===========================================================================
# Main pipeline
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("  EMPLOYEE ATTRITION PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load dataset ──────────────────────────────────────────
    logger.info("Step 1/5 → Loading dataset …")
    df = load_dataset(DATASET_PATH)
    if df is None:
        return

    # ── Step 2: Preprocessing ─────────────────────────────────────────
    logger.info("Step 2/5 → Preprocessing …")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.run(df)

    # ── Step 3: Train models ──────────────────────────────────────────
    logger.info("Step 3/5 → Training models …")
    trainer = ModelTrainer(feature_names)
    trainer.train_all(X_train, X_test, y_train, y_test)
    trainer.select_best()
    trainer.print_comparison_table()
    trainer.print_best_report(X_test, y_test)

    # ── Step 4: Save model ────────────────────────────────────────────
    logger.info("Step 4/5 → Saving best model …")
    trainer.save_best_model(MODEL_PATH)

    # ── Step 5: Explainability & evaluation plots ─────────────────────
    logger.info("Step 5/5 → Generating evaluation & SHAP plots …")
    generate_evaluation_plots(trainer, X_test, y_test)
    generate_shap_plots(
        trainer.best_model,
        trainer.best_model_name,
        X_test,
        feature_names,
    )

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Model  : {trainer.best_model_name}")
    print(f"  ROC-AUC     : {trainer.results[trainer.best_model_name]['ROC-AUC']}")
    print(f"  Saved to    : {MODEL_PATH}")
    print()
    print("  Generated files:")
    for f in [
        "eval_confusion_matrix.png",
        "eval_roc_curve.png",
        "eval_precision_recall.png",
        "shap_summary.png",
        "shap_feature_importance.png",
        "shap_waterfall.png",
    ]:
        print(f"    • {f}")
    print("\n  Next step → python predict.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
