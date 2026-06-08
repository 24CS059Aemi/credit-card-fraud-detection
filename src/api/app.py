"""Phase 7: Real-time fraud scoring API — sub-100ms response target."""
import time
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
app = Flask(__name__)

cfg = load_config()
api_cfg = cfg["api"]

model     = joblib.load(api_cfg["model_path"])
scaler    = joblib.load(api_cfg["scaler_path"])
threshold = joblib.load(api_cfg["threshold_path"])
logger.info(f"Model loaded. Threshold = {threshold:.4f}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "threshold": round(threshold, 4)})


@app.route("/score", methods=["POST"])
def score():
    """
    Score a single transaction.

    Expected JSON body:
    {
        "features": [v1, v2, ..., v28, amount, time]   # 30 raw values
    }

    Returns:
    {
        "fraud_probability": 0.9823,
        "is_fraud": true,
        "latency_ms": 2.1
    }
    """
    start = time.time()
    data = request.get_json(force=True)

    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    features = np.array(data["features"]).reshape(1, -1)
    if features.shape[1] != 30:
        return jsonify({"error": f"Expected 30 features, got {features.shape[1]}"}), 400

    prob = float(model.predict_proba(features)[0][1])
    is_fraud = bool(prob >= threshold)
    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(f"Score request | prob={prob:.4f} | fraud={is_fraud} | {latency_ms}ms")
    return jsonify({
        "fraud_probability": round(prob, 4),
        "is_fraud": is_fraud,
        "latency_ms": latency_ms
    })


@app.route("/batch_score", methods=["POST"])
def batch_score():
    """
    Score multiple transactions at once.

    Expected JSON body:
    {
        "transactions": [[30 values], [30 values], ...]
    }
    """
    start = time.time()
    data = request.get_json(force=True)

    if "transactions" not in data:
        return jsonify({"error": "Missing 'transactions' key"}), 400

    X = np.array(data["transactions"])
    probs = model.predict_proba(X)[:, 1]
    results = [
        {"fraud_probability": round(float(p), 4), "is_fraud": bool(p >= threshold)}
        for p in probs
    ]
    latency_ms = round((time.time() - start) * 1000, 2)
    return jsonify({"results": results, "latency_ms": latency_ms})


if __name__ == "__main__":
    app.run(host=api_cfg["host"], port=api_cfg["port"], debug=api_cfg["debug"])
