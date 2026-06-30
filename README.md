# Employee Attrition Prediction using Machine Learning

> Predict whether an employee is likely to leave your organisation — powered by Logistic Regression, Random Forest, and XGBoost with SHAP explainability.

---

## Introduction

Employee turnover is one of the most costly challenges modern organisations face. Losing a skilled employee can cost anywhere from 50% to 200% of their annual salary when accounting for recruitment, onboarding, and lost productivity.

This project applies the complete machine learning lifecycle to the **IBM HR Analytics Employee Attrition Dataset** to build a production-quality attrition prediction system. The system not only predicts *who* might leave, but also *explains why* using SHAP (SHapley Additive exPlanations).

---

## Objectives

- Build and compare multiple ML models to predict employee attrition
- Implement a complete preprocessing pipeline (cleaning, encoding, scaling, feature engineering)
- Generate actionable EDA visualisations for HR stakeholders
- Provide explainable predictions via SHAP
- Enable terminal-based real-time predictions for new employees
- Produce a portfolio-ready, GitHub-publishable codebase

---

## Features

- **Multi-model comparison** — Logistic Regression, Random Forest, XGBoost
- **Automatic best-model selection** based on ROC-AUC score
- **SHAP explainability** — understand *why* each prediction was made
- **Rich EDA** — 12+ professional Matplotlib and interactive Plotly charts
- **Interactive predictor** — enter employee details in the terminal and get instant results
- **Top influencing features** shown per prediction
- **Joblib model persistence** — save and reload trained models
- **PEP 8 compliant**, modular, type-hinted, fully documented code

---

## Dataset

| Property | Value |
|---|---|
| Source | [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Rows | 1,470 |
| Columns | 35 |
| Target | `Attrition` (Yes / No) |
| Class balance | ~16% attrition (imbalanced) |

**How to obtain:**
1. Visit the Kaggle link above
2. Click **Download** and unzip the archive
3. Rename the file to `employee_attrition.csv`
4. Place it in the project root directory

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Static visualisations |
| `plotly` | Interactive visualisations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `xgboost` | Gradient-boosted trees |
| `shap` | Model explainability |
| `joblib` | Model serialisation |
| `scipy` | Statistical utilities |

---

## Machine Learning Workflow

```
Raw CSV Data
    │
    ▼
Data Cleaning (duplicates, missing values, zero-variance columns)
    │
    ▼
Outlier Detection & Clipping (IQR method)
    │
    ▼
Target Encoding (Yes→1, No→0)
    │
    ▼
Feature Engineering (3 new derived features)
    │
    ▼
Categorical Encoding (Label + One-Hot)
    │
    ▼
Train/Test Split (80/20, stratified)
    │
    ▼
Feature Scaling (StandardScaler)
    │
    ▼
Model Training (LR · RF · XGB)
    │
    ▼
Evaluation & Comparison (Accuracy · Precision · Recall · F1 · AUC)
    │
    ▼
Best Model Selection → Saved to best_model.pkl
    │
    ▼
SHAP Explainability (Summary · Feature Importance · Waterfall)
    │
    ▼
Real-time Prediction (predict.py)
```

---

## Algorithms Used

| Model | Key Hyperparameters | Class Imbalance Handling |
|---|---|---|
| Logistic Regression | `max_iter=1000` | `class_weight="balanced"` |
| Random Forest | `n_estimators=200`, `max_depth=10` | `class_weight="balanced"` |
| XGBoost | `n_estimators=200`, `learning_rate=0.05` | `scale_pos_weight=5` |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2. Create and Activate a Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place `employee_attrition.csv` in the project root.

---

## How to Run

### Exploratory Data Analysis

```bash
python eda.py
```

Generates 10+ PNG charts and 2 interactive HTML files in the current directory.

### Train Models

```bash
python train_model.py
```

Trains all three models, prints the comparison table, saves the best model to `best_model.pkl`, and generates SHAP and evaluation plots.

### Predict Attrition

```bash
python predict.py
```

Launches an interactive terminal session. Enter employee details and receive a prediction with confidence percentage and top influencing features.

---

## Expected Output

### Training Summary

```
══════════════════════════════════════════════════════════════
  MODEL COMPARISON TABLE
══════════════════════════════════════════════════════════════
  Model                  Accuracy  Precision   Recall       F1      AUC
──────────────────────────────────────────────────────────────
  Logistic Regression      0.8299     0.5714   0.5172   0.5430   0.8412
  Random Forest            0.8537     0.6364   0.4828   0.5490   0.8621 ⭐
  XGBoost                  0.8435     0.6087   0.4828   0.5385   0.8554
══════════════════════════════════════════════════════════════
```

### Prediction Output

```
══════════════════════════════════════════════════════════════
  PREDICTION RESULT
══════════════════════════════════════════════════════════════
  🔴  PREDICTION  :  Employee Likely to LEAVE

  Probability of Leaving  :  71.34%
  Probability of Staying  :  28.66%
  Confidence              :  71.34%
  Model Used              : Random Forest

  Top 5 Influencing Features (SHAP):
  ──────────────────────────────────────────────────────
  1. OverTime_Yes                      ↑ increases risk (+0.1823)
  2. MonthlyIncome                     ↓ reduces risk   (-0.1201)
  3. YearsAtCompany                    ↓ reduces risk   (-0.0934)
  4. JobSatisfaction                   ↓ reduces risk   (-0.0712)
  5. DistanceFromHome                  ↑ increases risk (+0.0601)
══════════════════════════════════════════════════════════════
```

---

## Project Structure

```
employee-attrition-prediction/
│
├── employee_attrition.py     # Core module: loading, preprocessing, training
├── eda.py                    # Exploratory Data Analysis visualisations
├── train_model.py            # Full training pipeline + SHAP plots
├── predict.py                # Interactive terminal prediction module
│
├── requirements.txt          # Pinned dependencies
├── README.md                 # Project documentation
├── GUIDE.md                  # Beginner-friendly setup guide
├── LICENSE                   # MIT License
├── .gitignore                # Files to exclude from Git
│
├── employee_attrition.csv    # IBM HR dataset (download separately)
└── best_model.pkl            # Saved best model (generated by train_model.py)
```

---

## Model Performance

*(Sample results — actual numbers may vary slightly due to random seed)*

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Accuracy | 0.83 | 0.85 | 0.84 |
| Precision | 0.57 | 0.64 | 0.61 |
| Recall | 0.52 | 0.48 | 0.48 |
| F1 Score | 0.54 | 0.55 | 0.54 |
| ROC-AUC | 0.84 | **0.86** | 0.86 |

---

## Explainable AI (SHAP)

SHAP values explain the contribution of each feature to a prediction:

| Plot | File | Description |
|---|---|---|
| Summary (beeswarm) | `shap_summary.png` | Distribution of SHAP values; direction + magnitude per feature |
| Feature Importance | `shap_feature_importance.png` | Mean \|SHAP\| per feature — global importance ranking |
| Waterfall | `shap_waterfall.png` | How each feature pushes a single prediction from the baseline |

---

## Screenshots

*(Add screenshots after running the scripts)*

- `eda_attrition_distribution.png`
- `eda_correlation_heatmap.png`
- `eval_roc_curve.png`
- `shap_summary.png`

---

## Future Improvements

- **Hyperparameter Tuning** — GridSearchCV / Optuna for optimal params
- **Streamlit Web App** — drag-and-drop HR data, visual dashboards
- **Flask / FastAPI REST API** — HR system integration endpoint
- **Docker containerisation** — one-command reproducible environment
- **Cloud Deployment** — AWS SageMaker, GCP Vertex AI, Azure ML
- **CI/CD Pipeline** — GitHub Actions for automated testing and retraining
- **Model Monitoring** — data drift detection with Evidently AI
- **SMOTE** — synthetic oversampling for better minority class recall

---

## Resume Highlights

```
• Built an end-to-end employee attrition prediction system in Python using
  Logistic Regression, Random Forest, and XGBoost; achieved 0.86 ROC-AUC.
• Implemented a complete ML pipeline: data cleaning, feature engineering,
  one-hot encoding, outlier detection, StandardScaler, and stratified splits.
• Integrated SHAP for model explainability, generating beeswarm, bar, and
  waterfall plots to communicate predictions to non-technical stakeholders.
• Developed an interactive terminal prediction module with real-time
  feature attribution and confidence scores.
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Stellar Intelligence Team**
Final-Year MCA (AI & Data Science)

ISRO Bharatiya Antariksh Hackathon 2025

---

*Built with ❤️ using Python, scikit-learn, XGBoost, and SHAP*
