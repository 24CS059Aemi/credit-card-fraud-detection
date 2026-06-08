"""Phase 3: Preprocessing, feature engineering, and train/val/test splitting."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def preprocess_and_split(df: pd.DataFrame, config: dict):
    """
    Scale Amount and Time, create stratified splits, apply SMOTE on train only.
    Returns: X_train_sm, X_val, X_test, y_train_sm, y_val, y_test, scaler
    """
    cfg = config["data"]
    preproc = config["preprocessing"]

    # Scale Amount and Time (V1-V28 are already PCA-scaled)
    scaler = StandardScaler()
    for col in preproc["features_to_scale"]:
        df[f"{col}_scaled"] = scaler.fit_transform(df[[col]])
        logger.info(f"Scaled column: {col}")

    # Drop original unscaled columns
    drop_cols = preproc["features_to_scale"] + [preproc["target_column"]]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[preproc["target_column"]]

    logger.info(f"Feature matrix shape: {X.shape}")

    # Stratified splits: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=cfg["val_size"],
        random_state=cfg["random_state"],
        stratify=y_temp
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    _log_split_ratios(y_train, y_val, y_test)

    # SMOTE on train only — never val or test
    sm_cfg = config["smote"]
    sm = SMOTE(random_state=sm_cfg["random_state"],
               sampling_strategy=sm_cfg["sampling_strategy"])
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    logger.info(f"After SMOTE — Train shape: {X_train_sm.shape}")
    logger.info(f"SMOTE class counts: {pd.Series(y_train_sm).value_counts().to_dict()}")

    # Save processed splits
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_train_sm, y_train_sm), processed_dir / "train.pkl")
    joblib.dump((X_val, y_val), processed_dir / "val.pkl")
    joblib.dump((X_test, y_test), processed_dir / "test.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    logger.info("Processed splits and scaler saved")

    return X_train_sm, X_val, X_test, y_train_sm, y_val, y_test, scaler


def _log_split_ratios(y_train, y_val, y_test):
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        fraud_pct = y.mean() * 100
        logger.info(f"{name} fraud rate: {fraud_pct:.4f}%  (n={len(y)})")
