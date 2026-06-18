# Gas Leak Detection

Machine learning pipeline for detecting and characterizing gas leaks using metal-oxide sensor arrays. Built on the [UCI Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset) (16 sensors, 10 batches, Jan 2008 – Feb 2011).

## What it does

The pipeline solves three tasks in sequence:

| Task | Type | Best result |
|------|------|-------------|
| **1. Leak detection** — is gas present? | Binary classification | Random Forest: **98.8% accuracy** (random split) |
| **2. Gas identification** — which gas? | Multi-class classification (6 classes) | Random Forest: **99.5% accuracy** (random split) |
| **3. Concentration estimation** — how much? | Regression (per gas) | Random Forest: **R² = 0.977** (random split) |

A drift analysis additionally evaluates how models trained on early batches degrade over time, demonstrating the sensor aging effect.

**Supported gases:** Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene

## Project structure

```
gas_leak_detection/
├── main.py                  # End-to-end pipeline entry point
├── demo_predict.py          # Quick demo: load a saved model and predict
├── config.py                # Paths, alarm thresholds, gas names
├── requirements.txt
├── src/
│   ├── data_loader.py       # UCI download + caching (Parquet)
│   ├── eda.py               # Exploratory data analysis & plots
│   ├── preprocessing.py     # Scaling, feature engineering
│   ├── models_classification.py  # Binary & multi-class models
│   ├── models_regression.py      # Regression models
│   ├── drift_analysis.py    # Batch-by-batch sensor drift evaluation
│   └── utils.py             # Shared helpers
├── data/
│   └── raw/                 # Downloaded dataset (git-ignored)
└── results/
    ├── figures/             # Generated plots (git-ignored)
    ├── metrics/             # CSV metric tables (git-ignored)
    └── models/              # Serialized models — .joblib (git-ignored)
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads dataset automatically on first run)
python main.py

# 3. Run a quick prediction demo
python demo_predict.py
```

Results land in `results/` — metrics as CSV, plots as PNG, trained models as `.joblib` files.

## Models compared

- Logistic Regression
- SVM (RBF kernel)
- Random Forest
- XGBoost

Each model is evaluated under two scenarios: **Scenario A** (random train/test split) and **Scenario B** (temporal batch split), which exposes the real-world impact of sensor drift.

## Alarm thresholds

Conservative early-warning levels based on OSHA/NIOSH guidelines:

| Gas | Threshold (ppmv) |
|-----|-----------------|
| Ethanol | 200 |
| Ethylene | 100 |
| Ammonia | 250 |
| Acetaldehyde | 150 |
| Acetone | 300 |
| Toluene | 50 |

## Dataset

UCI Machine Learning Repository — [Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset)  
Vergara et al., *Chemical Physics Letters*, 2012.
