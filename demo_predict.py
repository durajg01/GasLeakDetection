"""Demonstration: predict_alarm() on individual sensor readings.

Simulates how the system would behave on a new, unseen sensor reading.
Loads saved models from results/models/ (run main.py first to generate them).

Usage:
    python demo_predict.py
"""
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
)
logger = logging.getLogger(__name__)

from config import GAS_NAMES, MODELS_DIR
from src.data_loader import get_feature_cols, load_data
from src.preprocessing import random_split


def predict_alarm(
    sensor_reading: np.ndarray,
    model_alarm,
    model_gas,
    scaler,
) -> dict:
    """Predict alarm status and gas identity for a raw sensor reading.

    Parameters
    ----------
    sensor_reading : np.ndarray, shape (128,)
        Unscaled feature vector from the 16-sensor array.
    model_alarm : fitted binary classifier (predict / predict_proba)
    model_gas   : fitted multi-class classifier
    scaler      : fitted StandardScaler matching the training data

    Returns
    -------
    dict
        alarm              : bool — True if alarm threshold likely exceeded
        alarm_probability  : float — probability of alarm class
        predicted_gas      : str — name of detected gas
        status             : str — human-readable status message
    """
    x_s        = scaler.transform(sensor_reading.reshape(1, -1))
    alarm_pred = int(model_alarm.predict(x_s)[0])
    alarm_prob = (float(model_alarm.predict_proba(x_s)[0][1])
                  if hasattr(model_alarm, 'predict_proba') else float('nan'))
    gas_pred   = int(model_gas.predict(x_s)[0]) + 1   # back to 1-indexed

    return {
        'alarm':             bool(alarm_pred),
        'alarm_probability': round(alarm_prob, 3),
        'predicted_gas':     GAS_NAMES.get(gas_pred, f'Unknown ({gas_pred})'),
        'status':            ('ALARM — niebezpieczny poziom gazu!'
                              if alarm_pred else 'OK — poziom bezpieczny'),
    }


def _load_or_train(X_train, y_alarm_train, y_gas_train):
    """Return (alarm_model, gas_model) — load from disk or train quick fallbacks."""
    try:
        import joblib
        alarm_model = joblib.load(os.path.join(MODELS_DIR, 'binary_Random_Forest_A.pkl'))
        gas_model   = joblib.load(os.path.join(MODELS_DIR, 'multi_Random_Forest_A.pkl'))
        logger.info("Zaladowano zapisane modele z %s", MODELS_DIR)
    except FileNotFoundError:
        logger.warning("Nie znaleziono zapisanych modeli — trening szybkich modeli zastepczych.")
        logger.warning("Uruchom 'python main.py' aby wygenerowac pelne modele.")
        from sklearn.ensemble import RandomForestClassifier
        from config import RANDOM_STATE
        alarm_model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        alarm_model.fit(X_train, y_alarm_train)
        gas_model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        gas_model.fit(X_train, y_gas_train - 1)   # 0-indexed for consistency
    return alarm_model, gas_model


def main() -> None:
    logger.info("Ladowanie danych...")
    df = load_data(use_cache=True)

    (X_tA, X_eA, yg_tA, yg_eA,
     ya_tA, ya_eA, yc_tA, yc_eA, scaler) = random_split(df)

    alarm_model, gas_model = _load_or_train(X_tA, ya_tA, yg_tA)

    # Pick 5 random rows from the original (unscaled) data
    feature_cols = get_feature_cols(df)
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(df), 5, replace=False)

    print("\n" + "=" * 70)
    print("  DEMO: predict_alarm() na 5 losowych probkach")
    print("=" * 70)
    print(f"{'Idx':>5}  {'Prawdziwy gaz':<14} {'Konc.':>7}  {'Alarm':>5}  |  Predykcja")
    print("-" * 70)

    for idx in indices:
        row       = df.iloc[int(idx)]
        x_raw     = row[feature_cols].values.astype(float)
        true_gas  = row['gas_name']
        true_conc = row['concentration_ppmv']
        true_alrm = bool(row['alarm'])

        result = predict_alarm(x_raw, alarm_model, gas_model, scaler)
        correct = '✓' if result['alarm'] == true_alrm else '✗'

        print(f"{idx:>5}  {true_gas:<14} {true_conc:>7.1f}  {str(true_alrm):>5}  |  "
              f"{result['status']}  [{result['predicted_gas']}, "
              f"P(alarm)={result['alarm_probability']:.2f}] {correct}")

    print("=" * 70)


if __name__ == '__main__':
    main()
