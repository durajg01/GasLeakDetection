"""Train and evaluate regression models for Task 3 (concentration estimation).

A separate model is trained per gas class because concentration ranges differ
dramatically between gases (Toluene: 10–100 ppmv vs Acetone: 12–1000 ppmv).
A single shared model would struggle to learn all scales simultaneously.

Run standalone:  python -m src.models_regression
"""
import copy
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.utils import save_bar_comparison, save_metrics_csv

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False


def _regression_models() -> list:
    models = [
        ('Ridge Regression', Ridge(alpha=1.0)),
        ('Random Forest',    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    if _XGB:
        models.append(('XGBoost', xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0)))
    else:
        models.append(('Gradient Boosting',
                        GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)))
    return models


def run_regression(
    X_trainA, y_gas_trainA, y_conc_trainA,
    X_testA,  y_gas_testA,  y_conc_testA,
    X_trainB, y_gas_trainB, y_conc_trainB,
    X_testB,  y_gas_testB,  y_conc_testB,
) -> pd.DataFrame:
    """Train per-gas regression models for both split scenarios.

    Parameters
    ----------
    X_train* / X_test* : scaled feature arrays
    y_gas_train*       : gas class labels (1-indexed)
    y_conc_train*      : concentration in ppmv

    Returns
    -------
    pd.DataFrame  with columns: model, scenario, MAE, RMSE, R2
    """
    results = []
    scenarios = [
        ('A-losowy', X_trainA, y_gas_trainA, y_conc_trainA, X_testA, y_gas_testA, y_conc_testA),
        ('B-batch',  X_trainB, y_gas_trainB, y_conc_trainB, X_testB, y_gas_testB, y_conc_testB),
    ]

    for scenario, X_tr, y_gas_tr, y_conc_tr, X_te, y_gas_te, y_conc_te in scenarios:
        logger.info("Regression — Scenario %s", scenario)
        for model_name, template in _regression_models():
            mae, rmse, r2, y_pred = _train_per_gas(
                template, X_tr, y_gas_tr, y_conc_tr,
                X_te, y_gas_te, y_conc_te, model_name, scenario,
            )
            logger.info("  %-22s [%s]  MAE=%.1f  RMSE=%.1f  R2=%.3f",
                        model_name, scenario, mae, rmse, r2)
            results.append({'model': model_name, 'scenario': scenario,
                            'MAE': mae, 'RMSE': rmse, 'R2': r2})

            if scenario == 'A-losowy':
                _scatter_plot(y_conc_te, y_pred, model_name, r2, rmse)

    df = pd.DataFrame(results)
    save_metrics_csv(results, 'task3_regression_metrics.csv')

    save_bar_comparison(df, 'R2', 'Zadanie 3 — R2 (regresja stezenia, per gaz)',
                        'R²', 'task3_r2.png', ylim=(0, 1.1))
    save_bar_comparison(df, 'RMSE', 'Zadanie 3 — RMSE (regresja stezenia, per gaz)',
                        'RMSE (ppmv)', 'task3_rmse.png')

    return df


def _train_per_gas(
    template, X_tr, y_gas_tr, y_conc_tr,
    X_te, y_gas_te, y_conc_te,
    model_name: str, scenario: str,
) -> tuple:
    """Fit one copy of template per gas class and aggregate predictions.

    Parameters
    ----------
    template   : unfitted sklearn regressor
    *_tr / *te : arrays for training and testing
    model_name : display name for logging/saving
    scenario   : 'A-losowy' or 'B-batch'

    Returns
    -------
    (mae, rmse, r2, y_pred_all)
    """
    y_pred_all = np.zeros(len(y_conc_te))

    for gas_id in range(6):   # 0-indexed
        tr_mask = (y_gas_tr - 1) == gas_id
        te_mask = (y_gas_te - 1) == gas_id
        if tr_mask.sum() < 5 or te_mask.sum() < 1:
            continue
        m = copy.deepcopy(template)
        m.fit(X_tr[tr_mask], y_conc_tr[tr_mask])
        y_pred_all[te_mask] = m.predict(X_te[te_mask])

        suffix = 'A' if scenario == 'A-losowy' else 'B'
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(m, os.path.join(MODELS_DIR,
                                     f"regr_{model_name.replace(' ', '_')}_gas{gas_id+1}_{suffix}.pkl"))

    mae  = mean_absolute_error(y_conc_te, y_pred_all)
    rmse = float(np.sqrt(mean_squared_error(y_conc_te, y_pred_all)))
    r2   = r2_score(y_conc_te, y_pred_all)
    return mae, rmse, r2, y_pred_all


def _scatter_plot(y_true, y_pred, model_name: str, r2: float, rmse: float) -> None:
    """Scatter plot of true vs predicted concentrations.

    Parameters
    ----------
    y_true     : true concentration values
    y_pred     : predicted concentration values
    model_name : used in title and filename
    r2, rmse   : metrics for annotation
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    maxv = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([0, maxv], [0, maxv], 'r--', linewidth=1.5, label='Idealna predykcja')
    ax.set_title(f'{model_name} — Rzeczywiste vs Predykcja (Scen. A)\n'
                 f'R²={r2:.3f}, RMSE={rmse:.1f} ppmv', fontsize=11)
    ax.set_xlabel('Rzeczywiste stezenie (ppmv)')
    ax.set_ylabel('Predykowane stezenie (ppmv)')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"task3_scatter_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Scatter plot saved: %s", path)


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    from src.data_loader import load_data
    from src.preprocessing import random_split, batch_split

    df = load_data()
    (X_tA, X_eA, yg_tA, yg_eA, ya_tA, ya_eA, yc_tA, yc_eA, _) = random_split(df)
    (X_tB, X_eB, yg_tB, yg_eB, ya_tB, ya_eB, yc_tB, yc_eB, _) = batch_split(df)

    df_r = run_regression(X_tA, yg_tA, yc_tA, X_eA, yg_eA, yc_eA,
                          X_tB, yg_tB, yc_tB, X_eB, yg_eB, yc_eB)
    print(df_r.to_string())
