"""Phase 4 & 5: Multi-model training, evaluation, threshold tuning."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                              roc_curve, fbeta_score, confusion_matrix,
                              classification_report)
from xgboost import XGBClassifier

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def build_models(config: dict) -> dict:
    """Instantiate all candidate models from config."""
    mc = config["models"]
    return {
        "Logistic Regression": LogisticRegression(**mc["logistic_regression"]),
        "Random Forest":       RandomForestClassifier(**mc["random_forest"]),
        "XGBoost":             XGBClassifier(**mc["xgboost"]),
    }


def train_and_evaluate(models: dict, X_train, y_train, X_val, y_val,
                       config: dict) -> dict:
    """Train every model and compare ROC-AUC on validation set."""
    results = {}
    for name, model in models.items():
        logger.info(f"Training: {name}")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        results[name] = {"model": model, "val_probs": probs, "val_auc": auc}
        logger.info(f"{name} — Val ROC-AUC: {auc:.4f}")

    # Summary table
    summary = pd.DataFrame({
        "Model": list(results.keys()),
        "Val ROC-AUC": [v["val_auc"] for v in results.values()]
    }).sort_values("Val ROC-AUC", ascending=False)
    logger.info(f"\n{summary.to_string(index=False)}")
    summary.to_csv("reports/model_comparison.csv", index=False)

    return results


def plot_roc_curves(results: dict, y_val,
                    save_path: str = "reports/figures/roc_curves.png"):
    plt.figure(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#F44336"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_val, res["val_probs"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['val_auc']:.4f})", color=color)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves — validation set")
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")


def tune_threshold(model, X_val, y_val, config: dict) -> float:
    """
    Sweep decision thresholds, pick the one that maximises F2-score
    (recall weighted 2x over precision — correct for fraud detection).
    """
    probs = model.predict_proba(X_val)[:, 1]
    cfg = config["evaluation"]
    thresholds = np.arange(cfg["threshold_range"][0],
                           cfg["threshold_range"][1],
                           cfg["threshold_step"])
    f2_scores = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        f2_scores.append(f2)

    best_idx = int(np.argmax(f2_scores))
    best_t = float(thresholds[best_idx])
    best_f2 = float(f2_scores[best_idx])

    logger.info(f"Best threshold: {best_t:.2f}  |  F2-score: {best_f2:.4f}")

    # Plot threshold vs F2
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, f2_scores, color="#FF7043")
    plt.axvline(best_t, color="red", linestyle="--", label=f"Best = {best_t:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("F2-score")
    plt.title("Threshold tuning (maximising F2-score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/threshold_tuning.png", dpi=150, bbox_inches="tight")
    plt.close()

    return best_t


def final_evaluation(model, X_test, y_test, threshold: float,
                     save_dir: str = "reports/figures"):
    """Evaluate best model on held-out test set with chosen threshold."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    f2  = fbeta_score(y_test, preds, beta=2, zero_division=0)
    cm  = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds,
                                   target_names=["Legitimate", "Fraud"])

    logger.info(f"TEST ROC-AUC : {auc:.4f}")
    logger.info(f"TEST F2-score: {f2:.4f}")
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"\n{report}")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    labels = [["TN", "FP"], ["FN", "TP"]]
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}",
                     ha="center", va="center", fontsize=12,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Predicted Legit", "Predicted Fraud"])
    plt.yticks([0, 1], ["Actual Legit", "Actual Fraud"])
    plt.title(f"Confusion matrix (threshold={threshold:.2f})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"test_roc_auc": auc, "test_f2": f2, "confusion_matrix": cm}


def save_best_model(model, threshold: float):
    """Persist model and threshold for the API."""
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    joblib.dump(threshold, "models/best_threshold.pkl")
    logger.info("Best model and threshold saved to models/")
