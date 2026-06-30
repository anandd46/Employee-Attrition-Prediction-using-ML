"""
predict.py
==========
Interactive Employee Attrition Prediction Module.

Allows users to enter employee information via the terminal and
receive a real-time prediction from the saved best model.

Outputs:
  - Prediction label (Stay / Leave)
  - Attrition probability
  - Confidence percentage
  - Top 5 features influencing the prediction (SHAP-based)

Run:
    python predict.py

Author  : Stellar Intelligence
Version : 1.0.0
"""

import logging
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap

from employee_attrition import MODEL_PATH, DataPreprocessor, load_dataset, load_model

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Input schema — prompts and validation rules
# ===========================================================================

# Each entry: (field_name, prompt_text, type, valid_options_or_range)
INPUT_SCHEMA: List[Tuple[str, str, type, Any]] = [
    ("Age", "Age (18-65)", int, (18, 65)),
    ("DailyRate", "Daily Rate (100-1500)", int, (100, 1500)),
    ("DistanceFromHome", "Distance from Home in km (1-30)", int, (1, 30)),
    ("Education", "Education Level (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor)", int, (1, 5)),
    ("EnvironmentSatisfaction", "Environment Satisfaction (1-4)", int, (1, 4)),
    ("HourlyRate", "Hourly Rate (30-100)", int, (30, 100)),
    ("JobInvolvement", "Job Involvement (1-4)", int, (1, 4)),
    ("JobLevel", "Job Level (1-5)", int, (1, 5)),
    ("JobSatisfaction", "Job Satisfaction (1-4)", int, (1, 4)),
    ("MonthlyIncome", "Monthly Income in USD (1000-20000)", int, (1000, 20000)),
    ("MonthlyRate", "Monthly Rate (2000-27000)", int, (2000, 27000)),
    ("NumCompaniesWorked", "Number of Companies Worked (0-9)", int, (0, 9)),
    ("PercentSalaryHike", "Percent Salary Hike last year (11-25)", int, (11, 25)),
    ("PerformanceRating", "Performance Rating (3=Excellent, 4=Outstanding)", int, (3, 4)),
    ("RelationshipSatisfaction", "Relationship Satisfaction (1-4)", int, (1, 4)),
    ("StockOptionLevel", "Stock Option Level (0-3)", int, (0, 3)),
    ("TotalWorkingYears", "Total Working Years (0-40)", int, (0, 40)),
    ("TrainingTimesLastYear", "Training Times Last Year (0-6)", int, (0, 6)),
    ("WorkLifeBalance", "Work-Life Balance (1-4)", int, (1, 4)),
    ("YearsAtCompany", "Years at Company (0-40)", int, (0, 40)),
    ("YearsInCurrentRole", "Years in Current Role (0-18)", int, (0, 18)),
    ("YearsSinceLastPromotion", "Years Since Last Promotion (0-15)", int, (0, 15)),
    ("YearsWithCurrManager", "Years with Current Manager (0-17)", int, (0, 17)),
    # Categorical
    ("BusinessTravel", "Business Travel (Non-Travel / Travel_Rarely / Travel_Frequently)", str,
     ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]),
    ("Department", "Department (Human Resources / Research & Development / Sales)", str,
     ["Human Resources", "Research & Development", "Sales"]),
    ("EducationField", "Education Field (Human Resources / Life Sciences / Marketing / Medical / Other / Technical Degree)", str,
     ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]),
    ("Gender", "Gender (Male / Female)", str, ["Male", "Female"]),
    ("JobRole", "Job Role (Healthcare Representative / Human Resources / Laboratory Technician / "
                "Manager / Manufacturing Director / Research Director / Research Scientist / "
                "Sales Executive / Sales Representative)", str,
     ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
      "Manufacturing Director", "Research Director", "Research Scientist",
      "Sales Executive", "Sales Representative"]),
    ("MaritalStatus", "Marital Status (Single / Married / Divorced)", str,
     ["Single", "Married", "Divorced"]),
    ("OverTime", "Overtime (Yes / No)", str, ["Yes", "No"]),
]


# ===========================================================================
# Input collection
# ===========================================================================

def _prompt(field: str, prompt: str, dtype: type, validator: Any) -> Any:
    """
    Prompt the user for a single field value with validation.

    Parameters
    ----------
    field     : column name
    prompt    : human-readable prompt text
    dtype     : expected Python type (int or str)
    validator : tuple (min, max) for numerics, list for categoricals
    """
    while True:
        try:
            raw = input(f"  {prompt}: ").strip()
            if dtype == int:
                value = int(raw)
                lo, hi = validator
                if not (lo <= value <= hi):
                    print(f"    ⚠️  Please enter a value between {lo} and {hi}.")
                    continue
                return value
            else:
                # Categorical — case-insensitive match
                options: List[str] = validator
                match = next(
                    (o for o in options if o.lower() == raw.lower()), None
                )
                if match is None:
                    print(f"    ⚠️  Choose from: {', '.join(options)}")
                    continue
                return match
        except ValueError:
            print("    ⚠️  Invalid input. Please try again.")


def collect_employee_input() -> Dict[str, Any]:
    """
    Collect all employee attributes interactively from the terminal.

    Returns
    -------
    dict mapping column names to user-entered values.
    """
    print("\n" + "=" * 60)
    print("  ENTER EMPLOYEE DETAILS")
    print("  (Press Ctrl+C at any time to exit)")
    print("=" * 60 + "\n")

    data: Dict[str, Any] = {}
    for field, prompt, dtype, validator in INPUT_SCHEMA:
        data[field] = _prompt(field, prompt, dtype, validator)

    return data


# ===========================================================================
# Feature alignment
# ===========================================================================

def _align_features(raw: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
    """
    Transform raw employee input into the exact feature vector the
    trained model expects.

    The model was trained on a preprocessed + one-hot-encoded dataset.
    We replicate that transformation here by building a dummy DataFrame
    and passing it through the same preprocessing steps.

    Parameters
    ----------
    raw           : dict of raw field values from the user
    feature_names : list of column names from training

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    # Add constant columns that were in the original dataset but may be
    # needed before we drop them in the preprocessor
    raw["EmployeeCount"] = 1
    raw["Over18"] = "Y"
    raw["StandardHours"] = 80
    raw["EmployeeNumber"] = 9999
    raw["Attrition"] = "No"  # dummy target — will be dropped

    df_input = pd.DataFrame([raw])

    # Run preprocessing (the preprocessor re-fits, so for prediction we
    # load the training data to establish the encoding schema, then
    # transform the single row)
    preprocessor = DataPreprocessor()

    train_df = load_dataset()
    if train_df is None:
        raise RuntimeError("Cannot align features without the training dataset.")

    preprocessor.clean(train_df)
    train_df = preprocessor.clean(train_df)
    train_df = preprocessor.remove_outliers_iqr(train_df)
    train_df = preprocessor.encode_target(train_df)
    train_df = preprocessor.feature_engineering(train_df)

    # Combine single input row with training data, transform together,
    # then extract the first row (index 0 from input)
    df_input = preprocessor.encode_target(df_input)
    df_input = preprocessor.feature_engineering(df_input)

    # Encode categoricals using training data as reference
    combined = pd.concat([df_input, train_df], ignore_index=True)
    combined = preprocessor.encode_categoricals(combined)

    # Keep only the first row (user's input)
    user_row = combined.iloc[[0]].drop(columns=["Attrition"], errors="ignore")

    # Re-index to match training feature names (fill missing with 0)
    user_row = user_row.reindex(columns=feature_names, fill_value=0)

    # Scale
    preprocessor.scaler.fit(train_df.drop(columns=["Attrition"], errors="ignore")
                            .reindex(columns=feature_names, fill_value=0))
    return preprocessor.scaler.transform(user_row)


# ===========================================================================
# SHAP-based feature explanation
# ===========================================================================

def _top_shap_features(
    model,
    model_name: str,
    X_input: np.ndarray,
    feature_names: List[str],
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Compute SHAP values for the input and return the top-N features
    by absolute SHAP value.

    Returns list of (feature_name, shap_value) tuples.
    """
    try:
        if "Logistic" in model_name:
            # Use linear explainer for speed
            explainer = shap.LinearExplainer(model, X_input)
            sv = explainer.shap_values(X_input)
            if isinstance(sv, list):
                sv = sv[1]
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_input)
            if isinstance(sv, list):
                sv = sv[1]

        shap_vals = sv[0]
        pairs = sorted(
            zip(feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return pairs[:top_n]
    except Exception:
        return []


# ===========================================================================
# Prediction
# ===========================================================================

def predict_single(bundle: dict, raw: Dict[str, Any]) -> None:
    """
    Run prediction for a single employee and print the result.

    Parameters
    ----------
    bundle : loaded model bundle (model, model_name, feature_names)
    raw    : dict of raw employee attribute values
    """
    model = bundle["model"]
    model_name = bundle["model_name"]
    feature_names = bundle["feature_names"]

    print("\n  ⏳  Preparing prediction …")

    try:
        X = _align_features(raw, feature_names)
    except Exception as exc:
        logger.error(f"Feature alignment failed: {exc}")
        return

    # Predict
    pred_label = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    prob_leave = prob[1]
    prob_stay = prob[0]
    confidence = max(prob_leave, prob_stay) * 100

    # SHAP top features
    top_features = _top_shap_features(model, model_name, X, feature_names)

    # ── Display result ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)

    if pred_label == 1:
        print("  🔴  PREDICTION  :  Employee Likely to LEAVE")
    else:
        print("  🟢  PREDICTION  :  Employee Likely to STAY")

    print(f"\n  Probability of Leaving  : {prob_leave * 100:6.2f}%")
    print(f"  Probability of Staying  : {prob_stay * 100:6.2f}%")
    print(f"  Confidence              : {confidence:6.2f}%")
    print(f"  Model Used              : {model_name}")

    if top_features:
        print("\n  Top 5 Influencing Features (SHAP):")
        print("  " + "-" * 50)
        for rank, (feat, val) in enumerate(top_features, start=1):
            direction = "↑ increases risk" if val > 0 else "↓ reduces risk"
            print(f"  {rank}. {feat:<35} {direction} ({val:+.4f})")

    print("=" * 60 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("  EMPLOYEE ATTRITION PREDICTION — PREDICTION MODULE")
    print("=" * 60)

    # Load the saved model bundle
    bundle = load_model(MODEL_PATH)
    if bundle is None:
        print("\n  ❌  No trained model found. Please run first:")
        print("       python train_model.py\n")
        return

    while True:
        try:
            # Collect employee data
            raw = collect_employee_input()

            # Run prediction
            predict_single(bundle, raw)

            # Ask whether to predict another
            again = input("  Predict another employee? (yes / no): ").strip().lower()
            if again not in ("yes", "y"):
                print("\n  Thank you for using AgriSat Intelligence — Employee Attrition Predictor! 👋\n")
                break

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye! 👋\n")
            break


if __name__ == "__main__":
    main()
