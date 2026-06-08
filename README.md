# Credit Card Fraud Detector

> Domain: FinTech · Imbalanced Classification · SMOTE · ROC-AUC

Detect fraudulent transactions in a dataset of 284,807 credit card transactions where only **0.17% are fraud**. This project implements the complete 8-phase ML pipeline — from data audit to a real-time scoring API — following industry practices used in production FinTech systems.

---

## Results

| Model               | Val ROC-AUC | Test ROC-AUC |
|---------------------|-------------|--------------|
| Logistic Regression | ~0.97       | —            |
| Random Forest       | ~0.98       | —            |
| **XGBoost (best)**  | **0.9891**  | **0.9874**   |

**Target: ROC-AUC ≥ 0.95** ✅  
Optimal threshold tuned via F2-score (recall weighted 2× over precision).

---

## Project Structure

```
credit-card-fraud-detector/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths in one place
│
├── data/
│   ├── raw/                     # Put creditcard.csv here (not committed)
│   └── processed/               # Auto-generated train/val/test splits (.pkl)
│
├── models/                      # Saved model, scaler, threshold (auto-generated)
│
├── notebooks/
│   └── 01_full_pipeline.ipynb   # End-to-end walkthrough notebook
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Phase 1-2: Load, audit, EDA plots
│   │   └── preprocessor.py      # Phase 3: Scale, split, SMOTE
│   ├── models/
│   │   ├── trainer.py           # Phase 4-5: Train, evaluate, tune threshold
│   │   └── error_analysis.py    # Phase 6: Inspect FP/FN patterns
│   ├── api/
│   │   └── app.py               # Phase 7: Flask scoring API
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
│
├── reports/
│   ├── figures/                 # All plots saved here automatically
│   ├── model_comparison.csv     # ROC-AUC comparison table
│   └── error_analysis_insights.md
│
├── tests/
│   ├── test_preprocessor.py
│   └── test_api.py
│
├── main.py                      # Run full pipeline with one command
├── requirements.txt
└── README.md
```

---

## Dataset

**Credit Card Fraud Detection** — publicly available on Kaggle.

- 284,807 transactions over 2 days
- 492 fraudulent transactions (0.17%)
- Features V1–V28: PCA-transformed for privacy (no original feature names)
- Features Amount and Time: raw, scaled during preprocessing
- Target: `Class` — 0 = legitimate, 1 = fraud

**Download:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Place the file at `data/raw/creditcard.csv`.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detector.git
cd credit-card-fraud-detector
```

### 2. Create virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download `creditcard.csv` from Kaggle and place it at:
```
data/raw/creditcard.csv
```

---

## Usage

### Run the complete pipeline

```bash
python main.py
```

This runs all 8 phases in sequence and saves models, plots, and reports automatically.

### Run individual phases

```bash
python main.py --phase eda          # Phase 1-2: Load data and plot EDA
python main.py --phase preprocess   # Phase 3: Scale, split, SMOTE
python main.py --phase train        # Phase 4-5: Train models, tune threshold
python main.py --phase evaluate     # Phase 6-7: Error analysis, final metrics
```

### Start the scoring API

```bash
python -m src.api.app
```

The API runs at `http://localhost:5000`. Test it:

```bash
# Health check
curl http://localhost:5000/health

# Score a single transaction (30 feature values)
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.2, 0.3, ..., 150.0, 43200.0]}'
```

**Response:**
```json
{
  "fraud_probability": 0.9823,
  "is_fraud": true,
  "latency_ms": 2.1
}
```

### Run tests

```bash
pytest tests/ -v
```

---

## The 8-Phase Pipeline

### Phase 1 — Data audit
Load `creditcard.csv`, check shape, dtypes, missing values, duplicates. Log fraud rate (0.17%). This step catches data quality issues before any modelling.

### Phase 2 — EDA
Three key visual insights:
1. Fraud transaction amounts cluster below $200 — counterintuitive but consistent with card-testing behaviour.
2. Fraud has two peaks in the time distribution — suggesting automated testing scripts running at low-traffic hours.
3. V4, V11, V14, V17 show the highest mean absolute difference between fraud and legit — these are the most discriminative PCA components.

### Phase 3 — Preprocessing & SMOTE
- Scale `Amount` and `Time` with `StandardScaler` (V1–V28 already scaled).
- Stratified split: 70% train / 15% val / 15% test — fraud ratio preserved in each split.
- Apply **SMOTE only on training data** — generates synthetic fraud samples by interpolating between existing fraud points in feature space.

> **Why SMOTE only on train?** Applying SMOTE to val/test would leak synthetic fraud patterns into evaluation, making your metrics unrealistically optimistic.

### Phase 4 — Multi-model training
Train three models and compare ROC-AUC on the validation set:

| Model               | Why it's included                                          |
|---------------------|-------------------------------------------------------------|
| Logistic Regression | Fast, interpretable baseline — explains trade-offs clearly |
| Random Forest       | Handles non-linear patterns, robust to noise               |
| XGBoost             | State-of-the-art for tabular data — typically best performer|

### Phase 5 — Threshold tuning
Default threshold = 0.5, but this is wrong for fraud detection. A missed fraud (false negative) costs far more than a false alarm (false positive). Solution: sweep thresholds from 0.1 to 0.9 and maximise **F2-score** (beta=2), which weights recall twice as heavily as precision.

### Phase 6 — Error analysis
Inspect false negatives and false positives. Key questions:
- What does missed fraud look like? (Amount? Time? Feature values?)
- What makes the model flag a legit transaction as fraud?
- Which PCA components are most important?

Three hypotheses for the next iteration are written to `reports/error_analysis_insights.md`.

### Phase 7 — Deployment pipeline
Flask REST API with two endpoints:
- `GET /health` — liveness check
- `POST /score` — score a single transaction, returns probability + flag + latency
- `POST /batch_score` — score multiple transactions at once

Target: sub-100ms response time per transaction.

### Phase 8 — Business report
All metrics, plots, and insights are saved to `reports/`. The final summary:
- Best model, test ROC-AUC, confusion matrix at optimal threshold
- Error analysis insights and hypotheses
- What worked, what didn't, what to try next

---

## Key Design Decisions

**Why F2-score for threshold tuning, not F1?**  
In fraud detection, the cost of missing real fraud (FN) is much higher than the cost of blocking a legitimate transaction (FP). F2-score penalises false negatives more heavily, making it the right business metric here.

**Why not accuracy?**  
A model that predicts "no fraud" for every transaction achieves 99.83% accuracy. That is useless. ROC-AUC and F2-score are the right metrics for imbalanced problems.

**Why SMOTE instead of just class_weight?**  
Both are used: SMOTE oversamples the training set, and models are also initialised with `class_weight='balanced'`. Using both together gives more stable decision boundaries than either alone.

**Why chronological val/test split is not needed here?**  
Unlike time-series forecasting, this dataset does not have temporal autocorrelation in features — V1–V28 are PCA-transformed and stationary. Standard stratified random split is appropriate. For a production system with real feature engineering, a time-ordered split would be required.

---

## EDA Insights

1. **Fraud amounts are surprisingly small** — median fraud amount is ~$9 vs ~$22 for legitimate. This is consistent with card-testing: fraudsters make small test transactions before larger ones.

2. **Two fraud clusters by time** — fraud peaks at hours 0–2 and 24–26 (low-traffic periods), suggesting automated scripts running when monitoring is lighter.

3. **V14 is the single most discriminative feature** — the largest mean difference between fraud and legit across all PCA components. V4, V11, and V17 follow closely.

---

## What I Would Try Next

- **Feature engineering:** Add merchant category (not available in this dataset but critical in production), velocity features (transactions per card per hour), and geographic distance between consecutive transactions.
- **LightGBM:** Faster than XGBoost with similar or better performance on tabular data.
- **Isolation Forest / Autoencoder:** Unsupervised anomaly detection as an additional signal to blend with the supervised model.
- **Calibrated probabilities:** Use `CalibratedClassifierCV` to ensure the output probability is a true estimate, not just a ranking score.
- **Online learning:** A fraud detection system needs to adapt to new patterns. Incremental learning with concept drift detection would be essential in production.

---

## Internship Context

This project was built as part of the **Synkoc AI/ML Internship** Capstone (Week 6, Lesson 13). It demonstrates:
- End-to-end ML pipeline on a real-world imbalanced dataset
- Industry-standard techniques: SMOTE, threshold tuning, error analysis
- Production-ready code structure with configs, logging, tests, and a REST API
- Professional documentation and portfolio-ready GitHub repository

---

## Author

**Aemi Patel**
[GitHub](https://github.com/24CS059Aemi)
