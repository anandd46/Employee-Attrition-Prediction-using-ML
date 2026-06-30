# Complete Beginner's Guide
## Employee Attrition Prediction — Setup to GitHub

---

## Table of Contents

1. [Installing Python](#1-installing-python)
2. [Installing VS Code](#2-installing-vs-code)
3. [Download the Project](#3-download-the-project)
4. [Create a Virtual Environment](#4-create-a-virtual-environment)
5. [Activate the Virtual Environment](#5-activate-the-virtual-environment)
6. [Install Dependencies](#6-install-dependencies)
7. [Running the Project](#7-running-the-project)
8. [Training the Model](#8-training-the-model)
9. [Running Predictions](#9-running-predictions)
10. [Common Errors & Fixes](#10-common-errors--fixes)
11. [Upload to GitHub](#11-upload-to-github)
12. [Publish the Repository](#12-publish-the-repository)
13. [Resume Tips](#13-resume-tips)
14. [Future Improvements](#14-future-improvements)

---

## 1. Installing Python

Python is the programming language used in this project.

### Download Python

1. Open your web browser and go to: **https://www.python.org/downloads/**
2. Click the yellow **"Download Python 3.x.x"** button (latest stable version).
3. The installer (`.exe` on Windows, `.pkg` on macOS) will download.

### Install Python on Windows

1. Double-click the downloaded installer.
2. **IMPORTANT**: Check the box **"Add Python to PATH"** at the bottom before clicking Install Now. If you miss this step, Python won't work from the terminal.
3. Click **"Install Now"**.
4. Wait for installation to complete, then click **"Close"**.

### Install Python on macOS

1. Double-click the downloaded `.pkg` file.
2. Follow the prompts (Continue → Agree → Install).
3. Python will be installed in `/usr/local/bin/python3`.

### Install Python on Linux (Ubuntu/Debian)

Open a terminal and run:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

### Verify Installation

Open a terminal (or Command Prompt on Windows) and type:

```bash
python --version
```

You should see something like: `Python 3.11.5`

On Linux/macOS you may need:

```bash
python3 --version
```

---

## 2. Installing VS Code

VS Code is a free, powerful code editor that makes Python development easy.

### Download VS Code

1. Go to: **https://code.visualstudio.com/**
2. Click **"Download"** for your operating system.
3. Install it like any normal application.

### Install the Python Extension

1. Open VS Code.
2. Press `Ctrl + Shift + X` (or `Cmd + Shift + X` on macOS) to open Extensions.
3. Search for **"Python"**.
4. Click **Install** on the extension by Microsoft.

This extension gives you syntax highlighting, IntelliSense (autocomplete), debugging, and the ability to run Python files directly.

---

## 3. Download the Project

### Option A — Clone with Git (recommended)

If you have Git installed:

```bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### Option B — Download ZIP

1. Go to the GitHub repository page.
2. Click the green **"Code"** button.
3. Click **"Download ZIP"**.
4. Unzip the downloaded file somewhere convenient (e.g., your Desktop).

### Open in VS Code

1. Open VS Code.
2. Go to **File → Open Folder**.
3. Navigate to the project folder and click **Select Folder**.

You'll see all the project files in the Explorer panel on the left.

---

## 4. Create a Virtual Environment

A virtual environment is an isolated Python sandbox for this project. It ensures the libraries you install here don't conflict with other Python projects on your computer.

Think of it as a separate "room" just for this project.

Open a terminal inside VS Code: **Terminal → New Terminal**

### Windows

```bash
python -m venv .venv
```

**What this does:**
- `python` — runs Python
- `-m venv` — uses Python's built-in virtual environment module
- `.venv` — name of the folder that will contain the environment (the dot makes it hidden)

### macOS / Linux

```bash
python3 -m venv .venv
```

After running this command, a new folder called `.venv` will appear in your project directory.

---

## 5. Activate the Virtual Environment

Activating "enters" the virtual environment so all subsequent Python commands run inside it.

### Windows (Command Prompt)

```bash
.venv\Scripts\activate
```

### Windows (PowerShell)

```bash
.venv\Scripts\Activate.ps1
```

If you get a "running scripts is disabled" error in PowerShell, run this first:

```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### How to know it's activated

After activation, your terminal prompt will change to show `(.venv)` at the start:

```
(.venv) C:\Users\YourName\employee-attrition-prediction>
```

To deactivate later (when you're done working):

```bash
deactivate
```

---

## 6. Install Dependencies

### What is requirements.txt?

The file `requirements.txt` lists every Python library this project needs, along with the exact version numbers. This ensures everyone running the project uses the same versions, preventing bugs caused by version differences.

### Install Everything

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

**What each part means:**
- `pip` — Python's package installer (like an app store for Python libraries)
- `install` — the action (install packages)
- `-r requirements.txt` — "read the requirements from this file"

This will download and install:
- `pandas` — data manipulation
- `numpy` — numerical computing
- `matplotlib` — charts and graphs
- `plotly` — interactive charts
- `scikit-learn` — machine learning algorithms
- `xgboost` — gradient boosting
- `shap` — model explainability
- `joblib` — model saving/loading
- `scipy` — scientific computing

Installation may take 2–5 minutes depending on your internet speed.

---

## 7. Running the Project

The project has three main scripts. Run them in this order:

### Step 1: Exploratory Data Analysis

```bash
python eda.py
```

**What happens:**
- Loads the dataset
- Generates 10+ professional charts (saved as PNG files in the folder)
- Generates 2 interactive Plotly charts (saved as HTML files)
- Prints progress to the terminal

**Expected output:**
```
14:32:01  [INFO]  ✅ Dataset loaded → 1470 rows × 35 columns
14:32:01  [INFO]  Generating: Attrition Distribution
14:32:02  [INFO]  Saved → eda_attrition_distribution.png
...
✅ EDA complete! All charts saved in the current directory.
```

### Step 2: Train Models

```bash
python train_model.py
```

### Step 3: Make Predictions

```bash
python predict.py
```

---

## 8. Training the Model

```bash
python train_model.py
```

### What Happens Internally

1. **Dataset Loading** — reads `employee_attrition.csv`
2. **Preprocessing** — cleans data, removes duplicates, handles missing values, clips outliers, encodes categories, scales features
3. **Training** — trains all three models (LR, RF, XGB) on 80% of the data
4. **Evaluation** — tests each model on the remaining 20%, computes 5 metrics
5. **Comparison** — prints a side-by-side comparison table
6. **Best Model Selection** — automatically picks the highest ROC-AUC model
7. **Saving** — saves the best model to `best_model.pkl`
8. **Plots** — generates and saves confusion matrix, ROC curve, precision-recall curve, and three SHAP plots

### Where the Model is Saved

The trained model is saved as `best_model.pkl` in your project directory.

`.pkl` stands for "pickle" — Python's format for serialising (saving) objects to disk.

To load it in Python:
```python
import joblib
bundle = joblib.load("best_model.pkl")
model = bundle["model"]
```

---

## 9. Running Predictions

```bash
python predict.py
```

The script will load the saved model and ask you to enter employee details one by one.

### Example Session

```
══════════════════════════════════════════════════════════════
  ENTER EMPLOYEE DETAILS
══════════════════════════════════════════════════════════════

  Age (18-65): 28
  Daily Rate (100-1500): 800
  Distance from Home in km (1-30): 15
  Education Level (1=Below College ... 5=Doctor): 3
  ...
  Overtime (Yes / No): Yes

══════════════════════════════════════════════════════════════
  PREDICTION RESULT
══════════════════════════════════════════════════════════════
  🔴  PREDICTION  :  Employee Likely to LEAVE

  Probability of Leaving  :  71.34%
  Probability of Staying  :  28.66%
  Confidence              :  71.34%

  Top 5 Influencing Features:
  1. OverTime_Yes             ↑ increases risk (+0.1823)
  2. MonthlyIncome            ↓ reduces risk   (-0.1201)
  ...
══════════════════════════════════════════════════════════════
```

---

## 10. Common Errors & Fixes

### ❌ `'python' is not recognized as an internal or external command`

**Cause:** Python was not added to PATH during installation.

**Fix (Windows):**
1. Uninstall Python from Control Panel.
2. Reinstall it, this time checking **"Add Python to PATH"** before clicking Install.

Or manually add Python to PATH via System Environment Variables.

---

### ❌ `'pip' is not recognized`

**Fix:**
```bash
python -m pip install -r requirements.txt
```

On Linux/macOS:
```bash
python3 -m pip install -r requirements.txt
```

---

### ❌ `FileNotFoundError: employee_attrition.csv`

**Cause:** Dataset file is missing.

**Fix:**
1. Download from: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
2. Rename the file to `employee_attrition.csv`
3. Place it in the same folder as the Python scripts

---

### ❌ `ModuleNotFoundError: No module named 'xgboost'`

**Cause:** Dependencies not installed, or virtual environment not activated.

**Fix:**
```bash
# Activate virtual environment first
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate        # Windows

# Then install
pip install -r requirements.txt
```

---

### ❌ SHAP errors (TreeExplainer / compatibility)

**Fix:**
```bash
pip install shap==0.44.1 --force-reinstall
```

---

### ❌ XGBoost installation issues on macOS

```bash
brew install libomp
pip install xgboost
```

---

### ❌ `joblib.externals.loky.process_executor.TerminatedWorkerError`

**Cause:** System ran out of memory.

**Fix:** Reduce `n_estimators` in `employee_attrition.py`:
```python
"n_estimators": 50,  # Reduce from 200
```

---

### ❌ `PermissionError` when saving model

**Cause:** Another program has the file open, or you don't have write access.

**Fix:**
- Close any program that might have `best_model.pkl` open.
- On Linux/macOS: `chmod 755 best_model.pkl`

---

### ❌ Version conflicts

**Fix:** Create a fresh virtual environment:
```bash
deactivate
rm -rf .venv       # macOS/Linux
rmdir /s .venv     # Windows

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 11. Upload to GitHub

### Prerequisites

1. Create a free account at **https://github.com**
2. Install Git: **https://git-scm.com/downloads**
3. Configure Git with your details:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Create a New Repository on GitHub

1. Log in to GitHub.
2. Click the **"+"** icon (top right) → **"New repository"**.
3. Name it: `employee-attrition-prediction`
4. Leave it **Public**.
5. Do NOT initialise with README (we already have one).
6. Click **"Create repository"**.

### Push Your Code

Open a terminal in your project folder and run these commands one by one:

```bash
git init
```
*Initialises a new Git repository in the current folder. Creates a hidden `.git` folder that tracks changes.*

```bash
git add .
```
*Stages all files for the next commit. The `.` means "everything in this directory".*

```bash
git commit -m "Initial commit: Employee Attrition Prediction"
```
*Creates a snapshot (commit) of all staged files. The `-m` flag sets the commit message.*

```bash
git branch -M main
```
*Renames the default branch to `main` (GitHub's standard name).*

```bash
git remote add origin https://github.com/YOUR_USERNAME/employee-attrition-prediction.git
```
*Tells Git where to push code. Replace `YOUR_USERNAME` with your actual GitHub username.*

```bash
git push -u origin main
```
*Uploads your code to GitHub. The `-u origin main` sets the upstream so future `git push` commands work without extra arguments.*

---

## 12. Publish the Repository

After pushing, make your repository shine:

### Add a Description

1. Go to your repository on GitHub.
2. Click the ⚙️ gear icon next to **"About"** (right side).
3. Write a one-line description: *"Employee attrition prediction using ML — LR, RF, XGBoost with SHAP explainability"*
4. Add your website if you have one.
5. Click **Save Changes**.

### Add Topics

In the same panel, add topics:
`machine-learning`, `python`, `employee-attrition`, `xgboost`, `shap`, `scikit-learn`, `data-science`, `hr-analytics`

Topics improve discoverability.

### Add a License

1. Click **"Add file" → "Create new file"**.
2. Name it `LICENSE`.
3. Click **"Choose a license template"** → select **MIT License**.
4. Fill in your name and year.
5. Commit the file.

### Create a Release

1. Click **"Releases"** (right sidebar) → **"Create a new release"**.
2. Tag version: `v1.0.0`
3. Title: *"v1.0.0 — Initial Release"*
4. Describe what's included.
5. Click **"Publish Release"**.

### Enable Issues & Discussions

Go to **Settings → General → Features** and toggle on **Issues** and **Discussions**. These let others report bugs or ask questions.

---

## 13. Resume Tips

### How to Write This on Your Resume

**Under Projects section:**

```
Employee Attrition Prediction                          Python · ML · SHAP
GitHub: github.com/YOUR_USERNAME/employee-attrition-prediction

• Built an end-to-end ML pipeline predicting employee turnover with 0.86 ROC-AUC
  using Logistic Regression, Random Forest, and XGBoost on the IBM HR dataset.
• Implemented data preprocessing: missing value imputation, IQR-based outlier
  clipping, Label/One-Hot encoding, StandardScaler, and stratified train/test split.
• Integrated SHAP explainability, generating beeswarm, bar, and waterfall plots
  to communicate model behaviour to non-technical stakeholders.
• Developed an interactive CLI prediction module with real-time feature attribution
  and probability confidence scores.
```

### Skills Demonstrated

| Skill | How |
|---|---|
| Python | Primary language for all modules |
| Data preprocessing | `DataPreprocessor` class with 6 pipeline stages |
| EDA | 12+ charts with Matplotlib and Plotly |
| ML algorithms | Logistic Regression, Random Forest, XGBoost |
| Model evaluation | Accuracy, Precision, Recall, F1, ROC-AUC |
| Explainable AI | SHAP summary, feature importance, waterfall |
| OOP | Classes with type hints and docstrings |
| Software engineering | Modular code, PEP 8, logging, exception handling |
| Git & GitHub | Version control, professional repository |

### Common Interview Questions & Answers

**Q: Why did you choose ROC-AUC as the primary metric?**
A: "The dataset is imbalanced — only ~16% of employees left. Accuracy is misleading because a model that always predicts 'Stay' would score 84%. ROC-AUC measures the model's ability to rank employees correctly regardless of threshold, making it more appropriate for imbalanced classification."

**Q: What is SHAP and why did you use it?**
A: "SHAP (SHapley Additive exPlanations) uses game theory to assign each feature a contribution value for each individual prediction. I used it to make the model interpretable — HR managers can understand *why* an employee is flagged as a flight risk, not just *whether* they are."

**Q: How did you handle class imbalance?**
A: "I used `class_weight='balanced'` for Logistic Regression and Random Forest, which internally adjusts the loss function to penalise misclassification of the minority class more. For XGBoost, I set `scale_pos_weight=5` (ratio of negatives to positives). I also used stratified splitting to preserve the class ratio in both train and test sets."

**Q: What would you do to improve this project?**
A: "Hyperparameter tuning with Optuna, adding SMOTE for synthetic oversampling, wrapping it in a Streamlit web app, deploying via FastAPI, containerising with Docker, and setting up CI/CD with GitHub Actions for automated retraining when new data arrives."

---

## 14. Future Improvements

### Hyperparameter Tuning
Use `GridSearchCV` or `Optuna` to find optimal model parameters:
```python
from sklearn.model_selection import GridSearchCV
param_grid = {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15]}
gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="roc_auc")
gs.fit(X_train, y_train)
```

### Streamlit Web App
```bash
pip install streamlit
streamlit run app.py
```
A web dashboard where HR staff can upload CSV files and view predictions visually.

### Flask / FastAPI REST API
Expose predictions as an HTTP endpoint so your HR system can call it programmatically:
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict(employee: EmployeeModel):
    return {"prediction": model.predict([employee.features])[0]}
```

### Docker Containerisation
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "train_model.py"]
```
Run anywhere with: `docker build -t attrition . && docker run attrition`

### Cloud Deployment
- **AWS SageMaker** — managed ML training and serving
- **GCP Vertex AI** — AutoML + custom model deployment
- **Azure ML** — enterprise ML lifecycle management

### CI/CD Pipeline (GitHub Actions)
Automatically retrain and test the model whenever new data is pushed:
```yaml
# .github/workflows/retrain.yml
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: python train_model.py
```

### Model Monitoring
Use **Evidently AI** to detect data drift — when incoming employee data starts to look different from training data, triggering retraining.

---

*Guide written for the ISRO Bharatiya Antariksh Hackathon 2025 — Stellar Intelligence Team*
