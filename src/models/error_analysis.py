"""Phase 6: Error pattern analysis — inspect what the model gets wrong."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_error_analysis(model, X_val, y_val, threshold: float,
                       save_dir: str = "reports/figures"):
    """
    Compare false negatives (missed fraud) and false positives (falsely blocked)
    against correctly classified samples. Saves insights to reports/.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)

    y_val_arr = np.array(y_val)
    fn_mask = (y_val_arr == 1) & (preds == 0)  # Fraud missed
    fp_mask = (y_val_arr == 0) & (preds == 1)  # Legit blocked
    tp_mask = (y_val_arr == 1) & (preds == 1)  # Fraud caught

    logger.info(f"False negatives (missed fraud) : {fn_mask.sum()}")
    logger.info(f"False positives (legit blocked): {fp_mask.sum()}")
    logger.info(f"True positives  (fraud caught) : {tp_mask.sum()}")

    X_val_arr = X_val.values if hasattr(X_val, "values") else X_val
    col_names = (X_val.columns.tolist() if hasattr(X_val, "columns")
                 else [f"f{i}" for i in range(X_val_arr.shape[1])])

    fn_df = pd.DataFrame(X_val_arr[fn_mask], columns=col_names)
    fp_df = pd.DataFrame(X_val_arr[fp_mask], columns=col_names)
    tp_df = pd.DataFrame(X_val_arr[tp_mask], columns=col_names)

    # Amount comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if "Amount_scaled" in col_names:
        amt_col = "Amount_scaled"
    else:
        amt_col = col_names[0]

    for df_sub, label, color in [
        (fn_df, "Missed fraud (FN)", "#F44336"),
        (fp_df, "Falsely blocked (FP)", "#FF9800"),
        (tp_df, "Caught fraud (TP)", "#4CAF50"),
    ]:
        if len(df_sub) > 0:
            axes[0].hist(df_sub[amt_col], bins=40, alpha=0.5,
                         label=label, color=color)
    axes[0].set_title("Amount distribution by error type")
    axes[0].set_xlabel("Amount (scaled)")
    axes[0].legend()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=col_names)
        top10 = importances.sort_values(ascending=False).head(10)
        top10.plot(kind="barh", ax=axes[1], color="#2196F3")
        axes[1].set_title("Top 10 feature importances")
        axes[1].invert_yaxis()
        logger.info(f"Top 5 features: {top10.index[:5].tolist()}")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Write insights markdown
    _write_insights(fn_mask.sum(), fp_mask.sum(), tp_mask.sum())
    logger.info("Error analysis complete")


def _write_insights(fn_count: int, fp_count: int, tp_count: int):
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    Path("reports").mkdir(exist_ok=True)
    with open("reports/error_analysis_insights.md", "w") as f:
        f.write("# Error analysis insights\n\n")
        f.write(f"- **Recall (fraud caught):** {recall:.2%}\n")
        f.write(f"- **Precision:** {precision:.2%}\n")
        f.write(f"- **Missed fraud (FN):** {fn_count}\n")
        f.write(f"- **Falsely blocked legit (FP):** {fp_count}\n\n")
        f.write("## Hypotheses for next iteration\n\n")
        f.write("1. Missed fraud transactions may have amounts similar to legitimate "
                "transactions — adding merchant category features could help distinguish them.\n")
        f.write("2. False positives may cluster in certain time windows — "
                "time-based features at a finer granularity could reduce false alarms.\n")
        f.write("3. A two-stage model (coarse filter → fine classifier) may reduce FP "
                "rate while preserving recall.\n")
