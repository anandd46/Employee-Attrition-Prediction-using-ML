"""
eda.py
======
Exploratory Data Analysis for Employee Attrition Prediction.


Generates professional visualizations using Matplotlib and Plotly:
- Attrition distribution
- Departmental breakdown
- Gender analysis
- Age distribution
- Job roles
- Education level
- Monthly income
- Overtime analysis
- Job satisfaction
- Years at company
- Correlation heatmap
- Feature importance

Run:  python eda.py

Author  : Stellar Intelligence
Version : 1.0.0
"""

import logging
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from employee_attrition import DATASET_PATH, load_dataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend — saves to file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Colour palette used across charts
PALETTE = {
    "stay": "#2ecc71",
    "leave": "#e74c3c",
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "accent": "#f39c12",
    "dark": "#2c3e50",
    "light_bg": "#f8f9fa",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "DejaVu Sans",
})


# ===========================================================================
# Helper utilities
# ===========================================================================

def _save(fig: plt.Figure, filename: str) -> None:
    """Save a Matplotlib figure and close it."""
    fig.savefig(filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved → {filename}")


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Attrition as a readable string label."""
    d = df.copy()
    if d["Attrition"].dtype != object:
        d["Attrition"] = d["Attrition"].map({1: "Yes", 0: "No"})
    return d


# ===========================================================================
# Individual plot functions
# ===========================================================================

def plot_attrition_distribution(df: pd.DataFrame) -> None:
    """Pie chart + bar chart showing overall attrition split."""
    d = _prep_df(df)
    counts = d["Attrition"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Employee Attrition Distribution", fontsize=14, fontweight="bold")

    # Pie chart
    axes[0].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[PALETTE["stay"], PALETTE["leave"]],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[0].set_title("Proportion")

    # Bar chart
    bars = axes[1].bar(
        counts.index,
        counts.values,
        color=[PALETTE["stay"], PALETTE["leave"]],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 5,
            str(val),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    axes[1].set_ylabel("Number of Employees")
    axes[1].set_title("Count by Category")
    axes[1].set_ylim(0, counts.max() * 1.15)

    plt.tight_layout()
    _save(fig, "eda_attrition_distribution.png")


def plot_department_analysis(df: pd.DataFrame) -> None:
    """Stacked bar of attrition rate per department."""
    d = _prep_df(df)
    dept_attr = (
        d.groupby(["Department", "Attrition"])
        .size()
        .unstack(fill_value=0)
    )
    dept_pct = dept_attr.div(dept_attr.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    dept_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[PALETTE["stay"], PALETTE["leave"]],
        edgecolor="white",
    )
    ax.set_title("Attrition Rate by Department", fontweight="bold")
    ax.set_xlabel("Department")
    ax.set_ylabel("Percentage (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(["Stay", "Leave"], loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    _save(fig, "eda_department_analysis.png")


def plot_gender_analysis(df: pd.DataFrame) -> None:
    """Side-by-side bar chart of gender vs attrition."""
    d = _prep_df(df)
    gender_attr = (
        d.groupby(["Gender", "Attrition"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    gender_attr.plot(
        kind="bar",
        ax=ax,
        color=[PALETTE["stay"], PALETTE["leave"]],
        edgecolor="white",
        width=0.6,
    )
    ax.set_title("Attrition by Gender", fontweight="bold")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Employees")
    ax.legend(["Stay", "Leave"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    _save(fig, "eda_gender_analysis.png")


def plot_age_distribution(df: pd.DataFrame) -> None:
    """Overlapping histograms comparing age of leavers vs stayers."""
    d = _prep_df(df)

    fig, ax = plt.subplots(figsize=(9, 4))
    for label, colour, alpha in [
        ("No", PALETTE["stay"], 0.6),
        ("Yes", PALETTE["leave"], 0.6),
    ]:
        subset = d[d["Attrition"] == label]["Age"]
        ax.hist(subset, bins=25, color=colour, alpha=alpha, edgecolor="white", label=label)

    ax.set_title("Age Distribution by Attrition", fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Frequency")
    ax.legend(title="Attrition")

    plt.tight_layout()
    _save(fig, "eda_age_distribution.png")


def plot_job_role(df: pd.DataFrame) -> None:
    """Horizontal bar chart of attrition count by job role."""
    d = _prep_df(df)
    role_attr = (
        d[d["Attrition"] == "Yes"]
        .groupby("JobRole")
        .size()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        role_attr.index, role_attr.values, color=PALETTE["leave"], edgecolor="white"
    )
    for bar, val in zip(bars, role_attr.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, str(val), va="center")

    ax.set_title("Attrition Count by Job Role", fontweight="bold")
    ax.set_xlabel("Number of Employees Who Left")

    plt.tight_layout()
    _save(fig, "eda_job_role.png")


def plot_education_level(df: pd.DataFrame) -> None:
    """Bar chart of attrition rate by education field."""
    d = _prep_df(df)
    edu_attr = (
        d.groupby(["EducationField", "Attrition"])
        .size()
        .unstack(fill_value=0)
    )
    edu_attr["Rate"] = edu_attr["Yes"] / (edu_attr["Yes"] + edu_attr["No"]) * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    edu_sorted = edu_attr["Rate"].sort_values(ascending=False)
    bars = ax.bar(
        edu_sorted.index, edu_sorted.values,
        color=PALETTE["secondary"], edgecolor="white"
    )
    for bar, val in zip(bars, edu_sorted.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.3,
            f"{val:.1f}%",
            ha="center",
            fontsize=9,
        )
    ax.set_title("Attrition Rate by Education Field", fontweight="bold")
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    _save(fig, "eda_education_level.png")


def plot_monthly_income(df: pd.DataFrame) -> None:
    """Box plot comparing monthly income of leavers vs stayers."""
    d = _prep_df(df)

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [
        d[d["Attrition"] == "No"]["MonthlyIncome"].values,
        d[d["Attrition"] == "Yes"]["MonthlyIncome"].values,
    ]
    bp = ax.boxplot(
        groups,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        widths=0.4,
    )
    for patch, colour in zip(bp["boxes"], [PALETTE["stay"], PALETTE["leave"]]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.8)

    ax.set_xticklabels(["Stay", "Leave"])
    ax.set_title("Monthly Income: Stay vs Leave", fontweight="bold")
    ax.set_ylabel("Monthly Income (USD)")

    plt.tight_layout()
    _save(fig, "eda_monthly_income.png")


def plot_overtime(df: pd.DataFrame) -> None:
    """Grouped bar chart: overtime vs no-overtime attrition rates."""
    d = _prep_df(df)
    ot_attr = (
        d.groupby(["OverTime", "Attrition"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ot_attr.plot(
        kind="bar",
        ax=ax,
        color=[PALETTE["stay"], PALETTE["leave"]],
        edgecolor="white",
        width=0.5,
    )
    ax.set_title("Attrition by Overtime Status", fontweight="bold")
    ax.set_xlabel("Overtime")
    ax.set_ylabel("Number of Employees")
    ax.legend(["Stay", "Leave"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    _save(fig, "eda_overtime.png")


def plot_job_satisfaction(df: pd.DataFrame) -> None:
    """Bar chart: attrition count across job satisfaction levels."""
    d = _prep_df(df)
    sat_labels = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    d["SatisfactionLabel"] = d["JobSatisfaction"].map(sat_labels)
    sat_attr = (
        d.groupby(["SatisfactionLabel", "Attrition"])
        .size()
        .unstack(fill_value=0)
        .reindex(["Low", "Medium", "High", "Very High"])
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sat_attr.plot(
        kind="bar", ax=ax,
        color=[PALETTE["stay"], PALETTE["leave"]],
        edgecolor="white", width=0.6,
    )
    ax.set_title("Attrition by Job Satisfaction Level", fontweight="bold")
    ax.set_xlabel("Job Satisfaction")
    ax.set_ylabel("Number of Employees")
    ax.legend(["Stay", "Leave"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    _save(fig, "eda_job_satisfaction.png")


def plot_years_at_company(df: pd.DataFrame) -> None:
    """KDE-style histogram of years at company coloured by attrition."""
    d = _prep_df(df)

    fig, ax = plt.subplots(figsize=(9, 4))
    for label, colour, alpha in [
        ("No", PALETTE["stay"], 0.6),
        ("Yes", PALETTE["leave"], 0.6),
    ]:
        subset = d[d["Attrition"] == label]["YearsAtCompany"]
        ax.hist(subset, bins=20, color=colour, alpha=alpha, edgecolor="white", label=label)

    ax.set_title("Years at Company — Stay vs Leave", fontweight="bold")
    ax.set_xlabel("Years at Company")
    ax.set_ylabel("Frequency")
    ax.legend(title="Attrition")

    plt.tight_layout()
    _save(fig, "eda_years_at_company.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap of numeric features."""
    d = df.copy()

    # Encode object columns so correlation can be computed
    for col in d.select_dtypes(include="object").columns:
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col].astype(str))

    numeric_df = d.select_dtypes(include=[np.number])
    # Keep top 15 most-correlated with Attrition for readability
    top_cols = (
        numeric_df.corr()["Attrition"]
        .abs()
        .sort_values(ascending=False)
        .head(15)
        .index.tolist()
    )
    corr = numeric_df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlation Coefficient")

    ax.set_xticks(range(len(top_cols)))
    ax.set_yticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_cols, fontsize=8)

    # Annotate cells
    for i in range(len(top_cols)):
        for j in range(len(top_cols)):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="black" if abs(corr.values[i, j]) < 0.7 else "white",
            )

    ax.set_title("Feature Correlation Heatmap (Top 15 vs Attrition)", fontweight="bold")

    plt.tight_layout()
    _save(fig, "eda_correlation_heatmap.png")


def plot_feature_importance(df: pd.DataFrame) -> None:
    """Train a quick RF and display top-20 feature importances."""
    d = df.copy()
    for col in d.select_dtypes(include="object").columns:
        le = LabelEncoder()
        d[col] = le.fit_transform(d[col].astype(str))

    X = d.drop(columns=["Attrition"], errors="ignore")
    y = d["Attrition"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top20 = importances.sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top20.index, top20.values, color=PALETTE["primary"], edgecolor="white")
    ax.set_title("Top 20 Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance Score")

    for bar, val in zip(bars, top20.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    _save(fig, "eda_feature_importance.png")


def plot_interactive_overview(df: pd.DataFrame) -> None:
    """
    Save an interactive Plotly HTML file with multiple subplots.
    Opens in any web browser.
    """
    d = _prep_df(df)

    # 1. Attrition by department (Plotly grouped bar)
    dept_fig = px.histogram(
        d,
        x="Department",
        color="Attrition",
        barmode="group",
        title="Department vs Attrition",
        color_discrete_map={"No": PALETTE["stay"], "Yes": PALETTE["leave"]},
    )
    dept_fig.write_html("eda_interactive_department.html")

    # 2. Monthly income vs age scatter coloured by attrition
    scatter_fig = px.scatter(
        d,
        x="Age",
        y="MonthlyIncome",
        color="Attrition",
        size="YearsAtCompany",
        hover_data=["JobRole", "Department"],
        title="Age vs Monthly Income (size = Years at Company)",
        color_discrete_map={"No": PALETTE["stay"], "Yes": PALETTE["leave"]},
        opacity=0.7,
    )
    scatter_fig.write_html("eda_interactive_scatter.html")

    logger.info("  Interactive Plotly charts saved as HTML files.")


# ===========================================================================
# Main entry point
# ===========================================================================

def run_eda() -> None:
    """Execute the complete EDA pipeline."""
    print("\n" + "=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    df = load_dataset()
    if df is None:
        return

    # Encode Attrition for numeric operations
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).fillna(df["Attrition"])

    steps = [
        ("Attrition Distribution", plot_attrition_distribution),
        ("Department Analysis", plot_department_analysis),
        ("Gender Analysis", plot_gender_analysis),
        ("Age Distribution", plot_age_distribution),
        ("Job Role", plot_job_role),
        ("Education Level", plot_education_level),
        ("Monthly Income", plot_monthly_income),
        ("Overtime", plot_overtime),
        ("Job Satisfaction", plot_job_satisfaction),
        ("Years at Company", plot_years_at_company),
        ("Correlation Heatmap", plot_correlation_heatmap),
        ("Feature Importance", plot_feature_importance),
        ("Interactive Charts (Plotly)", plot_interactive_overview),
    ]

    for step_name, fn in steps:
        logger.info(f"  Generating: {step_name}")
        try:
            fn(df)
        except Exception as exc:
            logger.warning(f"  ⚠️  {step_name} failed: {exc}")

    print("\n✅ EDA complete! All charts saved in the current directory.")
    print("   PNG files: open with any image viewer.")
    print("   HTML files: open with any web browser (interactive).")
    print()


if __name__ == "__main__":
    run_eda()
