"""Phase 1 & 2: Data loading, audit, and imbalance analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load raw CSV dataset and log a summary audit."""
    path = config["data"]["raw_path"]
    logger.info(f"Loading dataset from: {path}")

    df = pd.read_csv(path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"Duplicates: {df.duplicated().sum()}")

    class_counts = df["Class"].value_counts()
    fraud_pct = df["Class"].mean() * 100
    logger.info(f"Class distribution:\n{class_counts}")
    logger.info(f"Fraud rate: {fraud_pct:.4f}%")

    return df


def plot_class_imbalance(df: pd.DataFrame, save_path: str = "reports/figures/class_imbalance.png"):
    """Bar chart showing the extreme class imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts = df["Class"].value_counts()
    axes[0].bar(["Legitimate", "Fraud"], counts.values, color=["#2196F3", "#F44336"])
    axes[0].set_title("Transaction count by class")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=10)

    pcts = df["Class"].value_counts(normalize=True) * 100
    axes[1].pie(pcts.values, labels=["Legitimate", "Fraud"],
                colors=["#2196F3", "#F44336"], autopct="%1.3f%%", startangle=90)
    axes[1].set_title("Class distribution (%)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Class imbalance plot saved to {save_path}")


def plot_eda(df: pd.DataFrame, save_dir: str = "reports/figures"):
    """EDA plots: amount distribution, time distribution, correlation heatmap."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Amount distribution by class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cls, label, color in [(0, "Legitimate", "#2196F3"), (1, "Fraud", "#F44336")]:
        axes[0].hist(df[df["Class"] == cls]["Amount"], bins=60,
                     alpha=0.6, label=label, color=color)
    axes[0].set_title("Transaction amount by class")
    axes[0].set_xlabel("Amount (USD)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Time distribution by class
    for cls, label, color in [(0, "Legitimate", "#2196F3"), (1, "Fraud", "#F44336")]:
        axes[1].hist(df[df["Class"] == cls]["Time"] / 3600, bins=48,
                     alpha=0.6, label=label, color=color)
    axes[1].set_title("Transaction time by class (hours)")
    axes[1].set_xlabel("Hours since first transaction")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/amount_time_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("EDA plots saved")

    # Top discriminative features
    fraud = df[df["Class"] == 1].drop("Class", axis=1).mean()
    legit = df[df["Class"] == 0].drop("Class", axis=1).mean()
    diff = (fraud - legit).abs().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    diff.plot(kind="bar", color="#FF7043")
    plt.title("Top 10 features: mean absolute difference (fraud vs legit)")
    plt.ylabel("Absolute mean difference")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_diff.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Top discriminative features: {diff.index.tolist()}")
