
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_dataset


# ==============================================
# PROJECT PATHS
# ==============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

TARGET_ORDER = ["Dropout", "Enrolled", "Graduate"]


def run_eda():

    print("\n===================================")
    print("ACADEMICRISK AI - EDA ANALYSIS")
    print("===================================")

    df = load_dataset()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # ==========================================
    # 1. TARGET DISTRIBUTION
    # ==========================================

    print("\n[1] Target Distribution")

    counts = (
        df["Target"]
        .value_counts()
        .reindex(TARGET_ORDER, fill_value=0)
    )

    percentages = (
        counts / len(df) * 100
    ).round(2)

    target_report = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print(target_report)

    target_report.to_csv(
        TABLES_DIR / "target_distribution.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        x=counts.index,
        y=counts.values,
        color="steelblue",
        ax=ax
    )

    for i, count in enumerate(counts.values):
        ax.text(
            i,
            count + 25,
            f"{count}",
            ha="center",
            fontsize=11
        )

    ax.set_title("Academic Outcome Distribution")
    ax.set_xlabel("Academic Outcome")
    ax.set_ylabel("Number of Students")
    ax.set_ylim(0, counts.max() * 1.15)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "01_target_distribution.png",
        dpi=200
    )

    plt.close(fig)

    # ==========================================
    # 2. AGE DISTRIBUTION
    # ==========================================

    print("\n[2] Age at Enrollment")

    age_summary = (
        df.groupby("Target")["Age at enrollment"]
        .agg(["count", "mean", "median", "min", "max"])
        .reindex(TARGET_ORDER)
        .round(2)
    )

    print(age_summary)

    age_summary.to_csv(
        TABLES_DIR / "age_by_target.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Target",
        y="Age at enrollment",
        order=TARGET_ORDER,
        color="skyblue",
        ax=ax
    )

    ax.set_title("Age at Enrollment by Academic Outcome")
    ax.set_xlabel("Academic Outcome")
    ax.set_ylabel("Age at Enrollment")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "02_age_by_target.png",
        dpi=200
    )

    plt.close(fig)


    # ==========================================
    # 3. ADMISSION GRADE ANALYSIS
    # ==========================================

    print("\n[3] Admission Grade Analysis")

    admission_summary = (
        df.groupby("Target")["Admission grade"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reindex(TARGET_ORDER)
        .round(2)
    )

    print(admission_summary)

    admission_summary.to_csv(
        TABLES_DIR / "admission_grade_by_target.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Target",
        y="Admission grade",
        order=TARGET_ORDER,
        color="lightgreen",
        ax=ax
    )

    ax.set_title(
        "Admission Grade by Academic Outcome"
    )
    ax.set_xlabel("Academic Outcome")
    ax.set_ylabel("Admission Grade")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "03_admission_grade_by_target.png",
        dpi=200
    )

    plt.close(fig)

    print(
        "[SAVED] 03_admission_grade_by_target.png"
    )
    print(
        "[SAVED] admission_grade_by_target.csv"
    )


    # ==========================================
    # 4. COURSE AND DROPOUT ANALYSIS
    # ==========================================

    print("\n[4] Course and Dropout Analysis")

    course_summary = (
        df.groupby("Course")
        .agg(
            total_students=("Target", "size"),
            dropout_students=(
                "Target",
                lambda x: (x == "Dropout").sum()
            )
        )
    )

    course_summary["dropout_rate"] = (
        course_summary["dropout_students"]
        / course_summary["total_students"]
        * 100
    ).round(2)

    course_summary = course_summary.sort_values(
        "dropout_rate",
        ascending=False
    )

    print(course_summary.to_string())

    course_summary.to_csv(
        TABLES_DIR / "course_dropout_analysis.csv"
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.barplot(
        data=course_summary.reset_index(),
        x="dropout_rate",
        y="Course",
        orient="h",
        color="steelblue",
        ax=ax
    )

    ax.set_title("Dropout Rate by Course")
    ax.set_xlabel("Dropout Rate (%)")
    ax.set_ylabel("Course Code")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "04_course_dropout_rate.png",
        dpi=200
    )

    plt.close(fig)

    print("[SAVED] 04_course_dropout_rate.png")
    print("[SAVED] course_dropout_analysis.csv")


    # ==========================================
    # 5. FIRST SEMESTER PERFORMANCE ANALYSIS
    # ==========================================

    print("\n[5] First Semester Performance Analysis")

    semester_features = [
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
    ]

    semester_summary = (
        df.groupby("Target")[semester_features]
        .agg(["mean", "median"])
        .reindex(TARGET_ORDER)
        .round(2)
    )

    print(semester_summary.to_string())

    semester_summary.to_csv(
        TABLES_DIR / "first_semester_summary.csv"
    )

    # Plot 1: Approved curricular units
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Target",
        y="Curricular units 1st sem (approved)",
        order=TARGET_ORDER,
        color="lightblue",
        ax=ax
    )

    ax.set_title(
        "First Semester Approved Units by Academic Outcome"
    )
    ax.set_xlabel("Academic Outcome")
    ax.set_ylabel("Approved Curricular Units")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "05_first_sem_approved.png",
        dpi=200
    )

    plt.close(fig)

    # Plot 2: First semester grade
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Target",
        y="Curricular units 1st sem (grade)",
        order=TARGET_ORDER,
        color="lightgreen",
        ax=ax
    )

    ax.set_title(
        "First Semester Grade by Academic Outcome"
    )
    ax.set_xlabel("Academic Outcome")
    ax.set_ylabel("First Semester Grade")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "06_first_sem_grade.png",
        dpi=200
    )

    plt.close(fig)

    print("[SAVED] first_semester_summary.csv")
    print("[SAVED] 05_first_sem_approved.png")
    print("[SAVED] 06_first_sem_grade.png")



    print("\n===================================")
    print("[SUCCESS] EDA completed.")
    print("===================================")

    print("\nSaved figures:")
    print("01_target_distribution.png")
    print("02_age_by_target.png")

    print("\nSaved tables:")
    print("target_distribution.csv")
    print("age_by_target.csv")


if __name__ == "__main__":
    run_eda()
