"""Data splitting and scaling for both train/test scenarios.

Scenario A — random stratified 80/20 split (optimistic baseline).
Scenario B — temporal split: train on batches 1-5, test on batches 6-10.

Scenario B simulates realistic deployment where the model is trained on older
data and tested on newer measurements that exhibit sensor drift.

Run standalone:  python -m src.preprocessing
"""
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from src.data_loader import get_feature_cols

logger = logging.getLogger(__name__)


def random_split(df: pd.DataFrame, test_size: float = 0.2):
    """Scenario A: random stratified split, scaler fit only on train.

    Parameters
    ----------
    df        : full dataset DataFrame
    test_size : fraction for test set

    Returns
    -------
    tuple of (X_train, X_test, y_gas_train, y_gas_test,
              y_alarm_train, y_alarm_test, y_conc_train, y_conc_test, scaler)
    All X arrays are StandardScaler-transformed.
    """
    feature_cols = get_feature_cols(df)
    X       = df[feature_cols].values
    y_gas   = df['gas_class'].values
    y_alarm = df['alarm'].values
    y_conc  = df['concentration_ppmv'].values

    (X_train, X_test,
     y_gas_tr, y_gas_te,
     y_alarm_tr, y_alarm_te,
     y_conc_tr, y_conc_te) = train_test_split(
        X, y_gas, y_alarm, y_conc,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_gas,
    )

    scaler = StandardScaler().fit(X_train)
    logger.info(
        "Scenario A (random): train=%d  test=%d  alarm_rate_train=%.2f  alarm_rate_test=%.2f",
        len(X_train), len(X_test),
        y_alarm_tr.mean(), y_alarm_te.mean(),
    )
    _kw = dict(nan=0.0, posinf=0.0, neginf=0.0)
    return (
        np.nan_to_num(scaler.transform(X_train), **_kw),
        np.nan_to_num(scaler.transform(X_test),  **_kw),
        y_gas_tr, y_gas_te,
        y_alarm_tr, y_alarm_te,
        y_conc_tr, y_conc_te,
        scaler,
    )


def batch_split(df: pd.DataFrame, train_batches=(1, 2, 3, 4, 5), test_batches=(6, 7, 8, 9, 10)):
    """Scenario B: temporal split by batch, scaler fit only on train.

    Parameters
    ----------
    df            : full dataset DataFrame
    train_batches : batch IDs used for training
    test_batches  : batch IDs used for testing

    Returns
    -------
    Same tuple structure as random_split.
    """
    feature_cols = get_feature_cols(df)

    tr_mask = df['batch_id'].isin(train_batches)
    te_mask = df['batch_id'].isin(test_batches)

    X_train = df[tr_mask][feature_cols].values
    X_test  = df[te_mask][feature_cols].values
    y_gas_tr   = df[tr_mask]['gas_class'].values
    y_gas_te   = df[te_mask]['gas_class'].values
    y_alarm_tr = df[tr_mask]['alarm'].values
    y_alarm_te = df[te_mask]['alarm'].values
    y_conc_tr  = df[tr_mask]['concentration_ppmv'].values
    y_conc_te  = df[te_mask]['concentration_ppmv'].values

    scaler = StandardScaler().fit(X_train)
    logger.info(
        "Scenario B (batch): train=%d (batches %s)  test=%d (batches %s)",
        len(X_train), list(train_batches), len(X_test), list(test_batches),
    )
    _kw = dict(nan=0.0, posinf=0.0, neginf=0.0)
    return (
        np.nan_to_num(scaler.transform(X_train), **_kw),
        np.nan_to_num(scaler.transform(X_test),  **_kw),
        y_gas_tr, y_gas_te,
        y_alarm_tr, y_alarm_te,
        y_conc_tr, y_conc_te,
        scaler,
    )


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    from src.data_loader import load_data
    df = load_data()

    splitA = random_split(df)
    splitB = batch_split(df)

    print(f"\nScenario A — train: {splitA[0].shape[0]}, test: {splitA[1].shape[0]}")
    print(f"  Alarm rate train: {splitA[4].mean():.3f}  test: {splitA[5].mean():.3f}")

    print(f"\nScenario B — train: {splitB[0].shape[0]}, test: {splitB[1].shape[0]}")
    print(f"  Alarm rate train: {splitB[4].mean():.3f}  test: {splitB[5].mean():.3f}")
