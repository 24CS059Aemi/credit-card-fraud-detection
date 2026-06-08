"""
main.py — Run the complete 8-phase Credit Card Fraud Detection pipeline.

Usage:
    python main.py               # full pipeline
    python main.py --phase eda   # run only EDA
    python main.py --phase train # run only training
"""
import argparse
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.data.loader import load_raw_data, plot_class_imbalance, plot_eda
from src.data.preprocessor import preprocess_and_split
from src.models.trainer import (build_models, train_and_evaluate,
                                 plot_roc_curves, tune_threshold,
                                 final_evaluation, save_best_model)
from src.models.error_analysis import run_error_analysis

logger = get_logger("main")


def run_pipeline(phase: str = "all"):
    config = load_config()
    Path("models").mkdir(exist_ok=True)

    if phase in ("all", "eda"):
        logger.info("=" * 50)
        logger.info("PHASE 1-2: Data loading & EDA")
        logger.info("=" * 50)
        df = load_raw_data(config)
        plot_class_imbalance(df)
        plot_eda(df)

    if phase in ("all", "preprocess"):
        logger.info("=" * 50)
        logger.info("PHASE 3: Preprocessing & SMOTE")
        logger.info("=" * 50)
        if "df" not in dir():
            df = load_raw_data(config)
        (X_train_sm, X_val, X_test,
         y_train_sm, y_val, y_test, scaler) = preprocess_and_split(df, config)

    if phase in ("all", "train"):
        logger.info("=" * 50)
        logger.info("PHASE 4-5: Training & threshold tuning")
        logger.info("=" * 50)
        import joblib
        if "X_train_sm" not in dir():
            X_train_sm, y_train_sm = joblib.load("data/processed/train.pkl")
            X_val,      y_val      = joblib.load("data/processed/val.pkl")
            X_test,     y_test     = joblib.load("data/processed/test.pkl")

        models  = build_models(config)
        results = train_and_evaluate(models, X_train_sm, y_train_sm, X_val, y_val, config)
        plot_roc_curves(results, y_val)

        best_name  = max(results, key=lambda k: results[k]["val_auc"])
        best_model = results[best_name]["model"]
        logger.info(f"Best model: {best_name} — AUC {results[best_name]['val_auc']:.4f}")

        best_threshold = tune_threshold(best_model, X_val, y_val, config)

    if phase in ("all", "evaluate"):
        logger.info("=" * 50)
        logger.info("PHASE 6-7: Error analysis & final evaluation")
        logger.info("=" * 50)
        import joblib
        if "best_model" not in dir():
            best_model     = joblib.load("models/best_model.pkl")
            best_threshold = joblib.load("models/best_threshold.pkl")
            X_val,  y_val  = joblib.load("data/processed/val.pkl")
            X_test, y_test = joblib.load("data/processed/test.pkl")

        run_error_analysis(best_model, X_val, y_val, best_threshold)
        metrics = final_evaluation(best_model, X_test, y_test, best_threshold)
        save_best_model(best_model, best_threshold)

        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Final Test ROC-AUC : {metrics['test_roc_auc']:.4f}")
        logger.info(f"Final Test F2-score: {metrics['test_f2']:.4f}")
        target = config["evaluation"]["target_roc_auc"]
        status = "PASSED" if metrics["test_roc_auc"] >= target else "BELOW TARGET"
        logger.info(f"Target ({target}) : {status}")
        logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "eda", "preprocess", "train", "evaluate"])
    args = parser.parse_args()
    run_pipeline(args.phase)
