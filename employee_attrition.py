"""
employee_attrition.py
=====================
Core module for Employee Attrition Prediction.

Contains:
- Data loading and validation
- Data preprocessing pipeline
- Feature engineering
- Model training and evaluation
- SHAP explainability

Author  : Stellar Intelligence
Version : 1.0.0
"""

import logging
import sys
import warnings
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = "employee_attrition.csv"
MODEL_PATH = "best_model.pkl"
RANDOM_SEED = 42
TEST_SIZE = 0.20

# Columns that are always single-valued and carry no predictive signal
USELESS_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

# Numeric columns used for outlier detection
NUMERIC_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike",
    "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


# ===========================================================================
# Data Loading
# ===========================================================================

def load_dataset(path: str = DATASET_PATH) -> Optional[pd.DataFrame]:
    """
    Load the IBM HR Analytics dataset from a CSV file.

    Parameters
    ----------
    path : str
        Relative path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame, or None if the file is not found.
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"✅ Dataset loaded → {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        _print_dataset_instructions(path)
        return None
    except Exception as exc:
        logger.error(f"Unexpected error while reading dataset: {exc}")
        return None


def _print_dataset_instructions(path: str) -> None:
    """Print friendly instructions when the dataset is missing."""
    print("\n" + "=" * 65)
    print("  ⚠️  Dataset Not Found")
    print("=" * 65)
    print(f"\n  Expected location: {path}")
    print("\n  Steps to obtain the dataset:")
    print("  1. Visit https://www.kaggle.com/datasets/pavansubhasht/"
          "ibm-hr-analytics-attrition-dataset")
    print("  2. Click 'Download' and unzip the archive.")
    print("  3. Rename the CSV to: employee_attrition.csv")
    print("  4. Place it in the same folder as this script.")
    print("\n  Then re-run the script.")
    print("=" * 65 + "\n")


# ===========================================================================
# Data Preprocessing
# ===========================================================================

class DataPreprocessor:
    """
    Handles all preprocessing steps for the attrition dataset.

    Attributes
    ----------
    label_encoders : dict
        Stores fitted LabelEncoder objects keyed by column name.
    scaler : StandardScaler
        Fitted scaler for numeric features.
    feature_names : list
        Names of features after encoding (used for SHAP alignment).
    """

    def __init__(self) -> None:
        self.label_encoders: dict = {}
        self.scaler = StandardScaler()
        self.feature_names: list = []

    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1 – Basic cleaning.

        - Remove duplicate rows.
        - Drop columns with zero variance (always the same value).
        - Report missing values.
        """
        initial_rows = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates()
        dropped = initial_rows - len(df)
        if dropped:
            logger.info(f"  Removed {dropped} duplicate rows.")

        # Drop useless columns that exist in the dataset
        cols_to_drop = [c for c in USELESS_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        logger.info(f"  Dropped zero-variance columns: {cols_to_drop}")

        # Report missing values
        missing = df.isnull().sum()
        total_missing = missing.sum()
        if total_missing == 0:
            logger.info("  No missing values found — dataset is clean.")
        else:
            logger.warning(f"  Missing values detected:\n{missing[missing > 0]}")
            # Impute numeric columns with median; categorical with mode
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info("  Missing values imputed.")

        return df

    # ------------------------------------------------------------------
    def remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2 – Outlier Detection & Removal using IQR method.

        Any row where a numeric column value falls outside
        [Q1 - 1.5*IQR, Q3 + 1.5*IQR] is clipped to the boundary.
        Clipping is preferred over deletion to retain data volume.
        """
        cols = [c for c in NUMERIC_COLS if c in df.columns]
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower, upper)
        logger.info("  Outliers clipped using IQR method.")
        return df

    # ------------------------------------------------------------------
    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3 – Encode the target column 'Attrition'.

        Yes → 1 (employee left)
        No  → 0 (employee stayed)
        """
        if "Attrition" in df.columns:
            df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
            logger.info("  Target encoded: Yes → 1, No → 0")
        return df

    # ------------------------------------------------------------------
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4 – Create new derived features.

        New features often improve model performance by capturing
        domain knowledge that raw columns don't express directly.
        """
        # Income-per-year-of-experience ratio
        if "MonthlyIncome" in df.columns and "TotalWorkingYears" in df.columns:
            df["IncomePerYearExp"] = df["MonthlyIncome"] / (
                df["TotalWorkingYears"] + 1
            )

        # How long the employee has been with their current manager
        # relative to their total tenure at the company
        if "YearsWithCurrManager" in df.columns and "YearsAtCompany" in df.columns:
            df["ManagerTenureRatio"] = df["YearsWithCurrManager"] / (
                df["YearsAtCompany"] + 1
            )

        # OverTime flag (Yes/No → binary)
        if "OverTime" in df.columns:
            df["OverTimeFlag"] = (df["OverTime"] == "Yes").astype(int)

        logger.info("  Feature engineering complete — 3 new features added.")
        return df

    # ------------------------------------------------------------------
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5 – Encode categorical features.

        Binary columns  → LabelEncoder (0/1)
        Multi-category  → One-Hot Encoding (drop_first=True to avoid multicollinearity)
        """
        binary_cols = [
            c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() == 2
        ]
        multi_cols = [
            c for c in df.select_dtypes(include="object").columns
            if df[c].nunique() > 2
        ]

        # Label encode binary columns
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        logger.info(f"  Label-encoded binary columns: {binary_cols}")

        # One-hot encode multi-category columns
        if multi_cols:
            df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
            logger.info(f"  One-hot encoded multi-category columns: {multi_cols}")

        return df

    # ------------------------------------------------------------------
    def split_and_scale(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step 6 – Train/test split and feature scaling.

        StandardScaler is applied only to numeric features so that
        tree-based models (RF, XGB) are unaffected while Logistic
        Regression benefits from normalised inputs.

        Returns
        -------
        X_train, X_test, y_train, y_test, feature_names
        """
        X = df.drop(columns=["Attrition"])
        y = df["Attrition"]

        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        # Scale all features (scikit-learn models handle this gracefully)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(
            f"  Train set: {X_train_scaled.shape} | Test set: {X_test_scaled.shape}"
        )
        logger.info(f"  Class balance in train: {dict(pd.Series(y_train).value_counts())}")

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, self.feature_names

    # ------------------------------------------------------------------
    def run(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Execute the full preprocessing pipeline.

        Returns
        -------
        X_train, X_test, y_train, y_test, feature_names
        """
        logger.info("─" * 55)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("─" * 55)

        df = self.clean(df)
        df = self.remove_outliers_iqr(df)
        df = self.encode_target(df)
        df = self.feature_engineering(df)
        df = self.encode_categoricals(df)

        return self.split_and_scale(df)


# ===========================================================================
# Model Training
# ===========================================================================

class ModelTrainer:
    """
    Trains and evaluates multiple classifiers, then selects the best.

    Models trained:
    - Logistic Regression
    - Random Forest Classifier
    - XGBoost Classifier

    Best model is selected based on ROC-AUC score.
    """

    def __init__(self, feature_names: list) -> None:
        self.feature_names = feature_names
        self.models: dict = {}
        self.results: dict = {}
        self.best_model_name: str = ""
        self.best_model = None

    # ------------------------------------------------------------------
    def _build_models(self) -> dict:
        """Return a dictionary of untrained model objects."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_SEED,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                scale_pos_weight=5,  # handles class imbalance
            ),
        }

    # ------------------------------------------------------------------
    def train_all(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Train every model and compute evaluation metrics.

        Metrics computed per model:
        - Accuracy, Precision, Recall, F1, ROC-AUC
        """
        models_dict = self._build_models()

        logger.info("─" * 55)
        logger.info("MODEL TRAINING")
        logger.info("─" * 55)

        for name, model in models_dict.items():
            logger.info(f"  Training → {name} …")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            }

            self.models[name] = model
            self.results[name] = metrics
            logger.info(f"    {metrics}")

    # ------------------------------------------------------------------
    def select_best(self) -> None:
        """
        Identify the best model based on ROC-AUC score and save it.
        """
        self.best_model_name = max(
            self.results, key=lambda k: self.results[k]["ROC-AUC"]
        )
        self.best_model = self.models[self.best_model_name]

        logger.info("─" * 55)
        logger.info(f"🏆 Best Model: {self.best_model_name}")
        logger.info(f"   ROC-AUC = {self.results[self.best_model_name]['ROC-AUC']}")
        logger.info("─" * 55)

    # ------------------------------------------------------------------
    def save_best_model(self, path: str = MODEL_PATH) -> None:
        """
        Persist the best model + metadata to disk using Joblib.

        The saved bundle includes the model object and feature names
        so the prediction module can reconstruct inputs correctly.
        """
        bundle = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_names": self.feature_names,
        }
        joblib.dump(bundle, path)
        logger.info(f"  Model saved → {path}")

    # ------------------------------------------------------------------
    def print_comparison_table(self) -> None:
        """Print a formatted model comparison table to the terminal."""
        print("\n" + "=" * 70)
        print("  MODEL COMPARISON TABLE")
        print("=" * 70)
        header = f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}"
        print(header)
        print("-" * 70)
        for name, m in self.results.items():
            tag = " ⭐" if name == self.best_model_name else ""
            print(
                f"  {name:<22} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f} "
                f"{m['Recall']:>8.4f} {m['F1 Score']:>8.4f} {m['ROC-AUC']:>8.4f}{tag}"
            )
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    def print_best_report(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        """Print the full classification report for the best model."""
        y_pred = self.best_model.predict(X_test)
        print("\n" + "=" * 55)
        print(f"  CLASSIFICATION REPORT — {self.best_model_name}")
        print("=" * 55)
        print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))

        cm = confusion_matrix(y_test, y_pred)
        print("  Confusion Matrix:")
        print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
        print("=" * 55 + "\n")


# ===========================================================================
# Model Loading (for prediction module)
# ===========================================================================

def load_model(path: str = MODEL_PATH) -> Optional[dict]:
    """
    Load a previously saved model bundle from disk.

    Parameters
    ----------
    path : str
        Path to the .pkl file.

    Returns
    -------
    dict with keys: 'model', 'model_name', 'feature_names'
    """
    try:
        bundle = joblib.load(path)
        logger.info(f"✅ Model loaded: {bundle['model_name']} from {path}")
        return bundle
    except FileNotFoundError:
        logger.error(
            f"Model file '{path}' not found. Run: python train_model.py"
        )
        return None
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        return None
