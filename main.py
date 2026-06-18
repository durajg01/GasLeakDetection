"""Gas Leak Detection — end-to-end ML pipeline.

Usage:
    python main.py

"""
import logging
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

# ─── logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ─── project imports ──────────────────────────────────────────────────────────
from config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_STATE, RESULTS_DIR
from src.data_loader import get_feature_cols, load_data
from src.drift_analysis import run_drift_analysis
from src.eda import run_eda
from src.models_classification import (
    run_binary_classification, run_multiclass_classification, tune_binary_rf,
)
from src.models_regression import run_regression
from src.preprocessing import batch_split, random_split

np.random.seed(RANDOM_STATE)


def main() -> None:
    logger.info("=" * 65)
    logger.info("  System wykrywania wyciekow gazu — pipeline ML")
    logger.info("=" * 65)

    for d in [FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Step 1: load data ─────────────────────────────────────────────────────
    logger.info("[1/7] Wczytanie danych (UCI Gas Sensor Array Drift, id=270)")
    df = load_data(use_cache=True)
    feature_cols = get_feature_cols(df)
    logger.info("Dataset: %d wierszy × %d cech  |  brakujace: %d",
                len(df), len(feature_cols), df.isnull().sum().sum())

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    logger.info("[2/7] Eksploracyjna analiza danych")
    run_eda(df)

    # ── Step 3: preprocessing ─────────────────────────────────────────────────
    logger.info("[3/7] Preprocessing i podzial danych")
    (X_tA, X_eA, yg_tA, yg_eA,
     ya_tA, ya_eA, yc_tA, yc_eA, scaler_A) = random_split(df)

    (X_tB, X_eB, yg_tB, yg_eB,
     ya_tB, ya_eB, yc_tB, yc_eB, scaler_B) = batch_split(df)

    # ── Step 4: Task 1 — binary alarm detection ───────────────────────────────
    logger.info("[4/7] Zadanie 1: Detekcja wycieku (klasyfikacja binarna)")
    df_binary = run_binary_classification(
        X_tA, ya_tA, X_eA, ya_eA,
        X_tB, ya_tB, X_eB, ya_eB,
    )
    logger.info("  Najlepszy recall (Scen. A): %.3f",
                df_binary[df_binary['scenario'] == 'A-losowy']['recall'].max())

    logger.info("[4b] Strojenie hiperparametrow RF (Zadanie 1)")
    tuning = tune_binary_rf(X_tA, ya_tA)
    logger.info("  Najlepszy recall CV: %.3f", tuning['best_cv_recall'])

    # ── Step 5: Task 2 — gas identification ───────────────────────────────────
    logger.info("[5/7] Zadanie 2: Identyfikacja gazu (klasyfikacja wieloklasowa)")
    df_multi = run_multiclass_classification(
        X_tA, yg_tA, X_eA, yg_eA,
        X_tB, yg_tB, X_eB, yg_eB,
        feature_names=feature_cols,
    )
    logger.info("  Najlepsza accuracy (Scen. A): %.3f",
                df_multi[df_multi['scenario'] == 'A-losowy']['accuracy'].max())

    # ── Step 6: Task 3 — concentration regression ─────────────────────────────
    logger.info("[6/7] Zadanie 3: Estymacja stezenia (regresja)")
    df_regr = run_regression(
        X_tA, yg_tA, yc_tA, X_eA, yg_eA, yc_eA,
        X_tB, yg_tB, yc_tB, X_eB, yg_eB, yc_eB,
    )
    logger.info("  Najlepsze R2 (Scen. A): %.3f",
                df_regr[df_regr['scenario'] == 'A-losowy']['R2'].max())

    # ── Step 7: drift analysis ────────────────────────────────────────────────
    logger.info("[7/7] Analiza dryftu sensorow")
    df_drift = run_drift_analysis(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(df_binary, df_multi, df_regr)
    _save_summary_md(df_binary, df_multi, df_regr, df_drift)

    logger.info("")
    logger.info("Pipeline zakonczony. Artefakty w katalogu: %s", RESULTS_DIR)


def _print_summary(df_binary, df_multi, df_regr) -> None:
    logger.info("")
    logger.info("=" * 65)
    logger.info("  PODSUMOWANIE WYNIKOW")
    logger.info("=" * 65)

    logger.info("\n[ZADANIE 1 — Detekcja wycieku (binarna)]")
    summary = (df_binary[['model', 'scenario', 'accuracy', 'recall', 'f1', 'roc_auc']]
               .sort_values(['scenario', 'recall'], ascending=[True, False]))
    logger.info("\n%s", summary.to_string(index=False))

    logger.info("\n[ZADANIE 2 — Identyfikacja gazu (multi-class)]")
    summary = (df_multi[['model', 'scenario', 'accuracy', 'macro_f1', 'weighted_f1']]
               .sort_values(['scenario', 'accuracy'], ascending=[True, False]))
    logger.info("\n%s", summary.to_string(index=False))

    logger.info("\n[ZADANIE 3 — Estymacja stezenia (regresja, per gaz)]")
    summary = (df_regr[['model', 'scenario', 'MAE', 'RMSE', 'R2']]
               .sort_values(['scenario', 'R2'], ascending=[True, False]))
    logger.info("\n%s", summary.to_string(index=False))


def _save_summary_md(df_binary, df_multi, df_regr, df_drift) -> None:
    """Write a human-readable Markdown summary to results/summary.md."""
    path = os.path.join(RESULTS_DIR, 'summary.md')
    lines = [
        "# Podsumowanie wynikow — System wykrywania wyciekow gazu\n",
        "## Zadanie 1 — Detekcja wycieku (klasyfikacja binarna)\n",
        df_binary[['model', 'scenario', 'accuracy', 'recall', 'f1', 'roc_auc']]
        .sort_values(['scenario', 'recall'], ascending=[True, False])
        .round(4).to_markdown(index=False),
        "\n\n## Zadanie 2 — Identyfikacja gazu (klasyfikacja wieloklasowa)\n",
        df_multi[['model', 'scenario', 'accuracy', 'macro_f1', 'weighted_f1']]
        .sort_values(['scenario', 'accuracy'], ascending=[True, False])
        .round(4).to_markdown(index=False),
        "\n\n## Zadanie 3 — Estymacja stezenia (regresja, per gaz)\n",
        df_regr[['model', 'scenario', 'MAE', 'RMSE', 'R2']]
        .sort_values(['scenario', 'R2'], ascending=[True, False])
        .round(4).to_markdown(index=False),
        "\n\n## Analiza dryftu (model trenowany na batch 1)\n",
        df_drift[['batch', 'n', 'alarm_f1', 'gas_f1']]
        .round(4).to_markdown(index=False),
        "\n",
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info("Summary report saved: %s", path)


if __name__ == '__main__':
    main()
