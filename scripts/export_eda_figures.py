"""Export the key EDA plots as standalone PNGs for the written report (reports/report.md).

The full interactive analysis lives in notebooks/01_exploratory_data_analysis.ipynb; this
script re-renders just the headline figures so the report doesn't depend on opening a notebook.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    CLEAN_DATA_FILE,
    CONTINUOUS_FEATURES,
    FIGURES_DIR,
    NOMINAL_FEATURES,
    RAW_DATA_FILE,
    TARGET,
)

sns.set_theme(style="whitegrid")


def plot_missing_values(raw_df: pd.DataFrame) -> None:
    missing = raw_df.isna().sum()
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    missing.plot(kind="bar", ax=ax, title="Missing values per column (raw data)", ylabel="count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_missing_values.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(clean_df: pd.DataFrame) -> None:
    counts = clean_df[TARGET].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(1)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str),
                palette=["#4C72B0", "#DD8452"], legend=False, ax=ax)
    for i, (count, p) in enumerate(zip(counts.values, pct.values, strict=True)):
        ax.text(i, count + 2, f"{count} ({p}%)", ha="center")
    ax.set_xlabel("target (0 = no disease, 1 = disease)")
    ax.set_ylabel("patients")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_continuous_distributions(clean_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), CONTINUOUS_FEATURES, strict=False):
        sns.histplot(data=clean_df, x=col, hue=TARGET, kde=True, bins=20, ax=ax,
                     palette=["#4C72B0", "#DD8452"], alpha=0.6, legend=False)
        ax.set_title(col)
    fig.suptitle("Continuous feature distributions by target (blue = no disease, orange = disease)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_continuous_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(clean_df: pd.DataFrame) -> None:
    corr = clean_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nominal_vs_target(clean_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(NOMINAL_FEATURES), figsize=(16, 4))
    for ax, col in zip(axes, NOMINAL_FEATURES, strict=True):
        rate = clean_df.groupby(col)[TARGET].mean().sort_index()
        sns.barplot(x=rate.index.astype(str), y=rate.values, hue=rate.index.astype(str),
                    palette="Blues_d", legend=False, ax=ax)
        ax.set_title(col)
        ax.set_ylabel("disease rate")
        ax.set_ylim(0, 1)
    fig.suptitle("Disease rate by nominal feature value")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_nominal_vs_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW_DATA_FILE)
    clean_df = pd.read_csv(CLEAN_DATA_FILE)

    plot_missing_values(raw_df)
    plot_class_balance(clean_df)
    plot_continuous_distributions(clean_df)
    plot_correlation_heatmap(clean_df)
    plot_nominal_vs_target(clean_df)
    print(f"Saved 5 EDA figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
