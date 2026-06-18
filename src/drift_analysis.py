"""Sensor drift experiment: train on batch 1, test on each subsequent batch.

This illustrates the well-known "sensor drift" / "concept drift" problem
described in Vergara et al. (2012).  Standard ML models degrade over time
as sensor characteristics change — without recalibration, transfer learning,
or domain adaptation, performance decays substantially by batch 10.

Run standalone:  python -m src.drift_analysis
"""
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from config import FIGURES_DIR, RANDOM_STATE
from src.data_loader import get_feature_cols
from src.utils import save_metrics_csv

logger = logging.getLogger(__name__)


def run_drift_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Train RF models on batch 1, evaluate on batches 2-10 sequentially.

    Parameters
    ----------
    df : full dataset DataFrame from load_data()

    Returns
    -------
    pd.DataFrame  with columns: batch, n, alarm_acc, alarm_f1, gas_acc, gas_f1
    """
    feature_cols = get_feature_cols(df)

    b1_mask    = df['batch_id'] == 1
    X_b1       = df[b1_mask][feature_cols].values
    y_b1_alarm = df[b1_mask]['alarm'].values
    y_b1_gas   = df[b1_mask]['gas_class'].values - 1   # 0-indexed for consistency

    scaler = StandardScaler().fit(X_b1)
    _kw    = dict(nan=0.0, posinf=0.0, neginf=0.0)
    X_b1_s = np.nan_to_num(scaler.transform(X_b1), **_kw)

    logger.info("Training drift models on batch 1 (%d samples)...", len(X_b1))
    rf_alarm = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_gas   = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_alarm.fit(X_b1_s, y_b1_alarm)
    rf_gas.fit(X_b1_s, y_b1_gas)

    records = []
    for bid in range(2, 11):
        mask = df['batch_id'] == bid
        if mask.sum() == 0:
            continue
        X_b    = np.nan_to_num(scaler.transform(df[mask][feature_cols].values), **_kw)
        y_alrm = df[mask]['alarm'].values
        y_gas  = df[mask]['gas_class'].values - 1

        record = {
            'batch':     bid,
            'n':         int(mask.sum()),
            'alarm_acc': accuracy_score(y_alrm, rf_alarm.predict(X_b)),
            'alarm_f1':  f1_score(y_alrm, rf_alarm.predict(X_b), zero_division=0),
            'gas_acc':   accuracy_score(y_gas, rf_gas.predict(X_b)),
            'gas_f1':    f1_score(y_gas, rf_gas.predict(X_b), average='macro', zero_division=0),
        }
        logger.info("  Batch %2d (n=%4d):  Alarm F1=%.3f  |  Gas Macro-F1=%.3f",
                    bid, record['n'], record['alarm_f1'], record['gas_f1'])
        records.append(record)

    df_drift = pd.DataFrame(records)
    save_metrics_csv(records, 'drift_metrics.csv')
    _plot_drift(df_drift)
    return df_drift


def _plot_drift(df_drift: pd.DataFrame) -> None:
    """Plot performance decay over batches and save to FIGURES_DIR.

    Parameters
    ----------
    df_drift : DataFrame returned by run_drift_analysis()
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    panels = [
        (axes[0], 'alarm_f1', 'Detekcja wycieku (binarna)',      'F1'),
        (axes[1], 'gas_f1',   'Identyfikacja gazu (multi-class)', 'Macro-F1'),
    ]
    for ax, col, title, ylabel in panels:
        ax.plot(df_drift['batch'], df_drift[col],
                marker='o', color='steelblue', linewidth=2, label='Test F1')
        ax.scatter([1], [1.0], marker='*', s=200, color='green', zorder=5,
                   label='Batch 1 (train, F1≈1.0)')
        ax.set_title(f'Dryft sensorow: degradacja modelu\n{title}', fontsize=11)
        ax.set_xlabel('Batch (~ czas zbierania danych)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(1, 11))
        ax.legend()

    plt.suptitle('Model wytrenowany na Batch 1, testowany na Batchach 2–10\n'
                 '(Random Forest — bez kompensacji dryftu)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'drift_over_time.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Drift chart saved: %s", path)


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    from src.data_loader import load_data
    df = load_data()
    df_drift = run_drift_analysis(df)
    print(df_drift.to_string(index=False))
