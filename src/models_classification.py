"""Train and evaluate classification models for Task 1 (binary alarm) and Task 2 (gas ID).

Run standalone:  python -m src.models_classification
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

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.utils import save_bar_comparison, save_confusion_matrix, save_metrics_csv, save_metrics_json

logger = logging.getLogger(__name__)

GAS_LABELS = ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene']

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False
    logger.info("XGBoost not available; using sklearn GradientBoosting instead")


# ─── model factories ──────────────────────────────────────────────────────────

def _binary_models() -> list:
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ('Random Forest',       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        ('SVM (RBF)',           SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
    ]
    if _XGB:
        models.append(('XGBoost',
                        xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                          eval_metric='logloss', verbosity=0)))
    else:
        models.append(('Gradient Boosting',
                        GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)))
    return models


def _multi_models() -> list:
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ('Random Forest',       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        ('SVM (RBF)',           SVC(kernel='rbf', random_state=RANDOM_STATE)),
    ]
    if _XGB:
        models.append(('XGBoost',
                        xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                          eval_metric='mlogloss', verbosity=0)))
    else:
        models.append(('Gradient Boosting',
                        GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)))
    return models


# ─── Task 1: binary alarm detection ──────────────────────────────────────────

def run_binary_classification(
    X_trainA, y_alarm_trainA, X_testA, y_alarm_testA,
    X_trainB, y_alarm_trainB, X_testB, y_alarm_testB,
) -> pd.DataFrame:
    """Train all binary classifiers for both split scenarios.

    The critical metric is **recall** (sensitivity to alarms): in a safety
    system a false negative (missed gas leak) is far more dangerous than a
    false positive.

    Parameters
    ----------
    X_trainA / X_testA : scaled feature arrays for Scenario A
    y_alarm_trainA / y_alarm_testA : binary alarm labels for Scenario A
    X_trainB / X_testB : scaled arrays for Scenario B
    y_alarm_trainB / y_alarm_testB : binary alarm labels for Scenario B

    Returns
    -------
    pd.DataFrame  with columns: model, scenario, accuracy, precision, recall, f1, roc_auc
    """
    results = []
    scenarios = [
        ('A-losowy', X_trainA, y_alarm_trainA, X_testA, y_alarm_testA),
        ('B-batch',  X_trainB, y_alarm_trainB, X_testB, y_alarm_testB),
    ]

    for scenario, X_tr, y_tr, X_te, y_te in scenarios:
        logger.info("Binary classification — Scenario %s", scenario)
        for name, template in _binary_models():
            m = copy.deepcopy(template)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            y_prob = m.predict_proba(X_te)[:, 1] if hasattr(m, 'predict_proba') else None

            row = {
                'model':     name,
                'scenario':  scenario,
                'accuracy':  accuracy_score(y_te, y_pred),
                'precision': precision_score(y_te, y_pred, zero_division=0),
                'recall':    recall_score(y_te, y_pred, zero_division=0),
                'f1':        f1_score(y_te, y_pred, zero_division=0),
                'roc_auc':   roc_auc_score(y_te, y_prob) if y_prob is not None else float('nan'),
            }
            logger.info("  %-28s [%s]  acc=%.3f  recall=%.3f  f1=%.3f  auc=%.3f",
                        name, scenario, row['accuracy'], row['recall'], row['f1'], row['roc_auc'])
            results.append(row)

            _save_model(m, f"binary_{name.replace(' ', '_')}_{scenario[0]}.pkl")

    df = pd.DataFrame(results)
    save_metrics_csv(results, 'task1_binary_metrics.csv')
    save_bar_comparison(df, 'recall', 'Zadanie 1 — Recall (detekcja wycieku)',
                        'Recall', 'task1_recall.png', ylim=(0, 1.1))
    save_bar_comparison(df, 'f1', 'Zadanie 1 — F1 (detekcja wycieku)',
                        'F1', 'task1_f1.png', ylim=(0, 1.1))

    best_name = (df[df['scenario'] == 'A-losowy']
                 .sort_values('recall', ascending=False).iloc[0]['model'])
    logger.info("Best binary model by recall (Scenario A): %s", best_name)

    for scenario, X_tr, y_tr, X_te, y_te in scenarios:
        m = copy.deepcopy(dict(_binary_models())[best_name])
        m.fit(X_tr, y_tr)
        save_confusion_matrix(
            y_te, m.predict(X_te),
            ['Brak alarmu', 'ALARM'],
            f'{best_name} — Scen. {scenario}',
            f'task1_confusion_{scenario}.png',
        )

    return df


def tune_binary_rf(X_train, y_train) -> dict:
    """GridSearchCV for Random Forest on the binary alarm task (maximises recall).

    Parameters
    ----------
    X_train : scaled training features
    y_train : binary alarm labels

    Returns
    -------
    dict with best_params and best_cv_recall
    """
    logger.info("GridSearchCV — RF binary (optimising recall)...")
    param_grid = {
        'n_estimators':    [100, 200],
        'max_depth':       [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    cv = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='recall',
        n_jobs=-1,
        verbose=0,
    )
    cv.fit(X_train, y_train)
    logger.info("Best params: %s  best CV recall: %.3f", cv.best_params_, cv.best_score_)

    _save_model(cv.best_estimator_, 'binary_RF_tuned.pkl')
    result = {'best_params': cv.best_params_, 'best_cv_recall': float(cv.best_score_)}
    save_metrics_json(result, 'task1_rf_tuning.json')
    return result


# ─── Task 2: multi-class gas identification ───────────────────────────────────

def run_multiclass_classification(
    X_trainA, y_gas_trainA, X_testA, y_gas_testA,
    X_trainB, y_gas_trainB, X_testB, y_gas_testB,
    feature_names: list | None = None,
) -> pd.DataFrame:
    """Train all multi-class classifiers for both split scenarios.

    Gas labels are shifted to 0-indexed internally for XGBoost compatibility
    and converted back for display.

    Parameters
    ----------
    X_trainA / X_testA  : scaled feature arrays for Scenario A
    y_gas_trainA / ...  : gas class labels (1-indexed) for Scenario A
    X_trainB / X_testB  : scaled arrays for Scenario B
    y_gas_trainB / ...  : gas class labels (1-indexed) for Scenario B
    feature_names       : optional list of feature column names for importance plot

    Returns
    -------
    pd.DataFrame  with columns: model, scenario, accuracy, macro_f1, weighted_f1
    """
    # XGBoost requires 0-indexed labels
    y_trainA_0, y_testA_0 = y_gas_trainA - 1, y_gas_testA - 1
    y_trainB_0, y_testB_0 = y_gas_trainB - 1, y_gas_testB - 1

    results = []
    scenarios = [
        ('A-losowy', X_trainA, y_trainA_0, X_testA, y_testA_0),
        ('B-batch',  X_trainB, y_trainB_0, X_testB, y_testB_0),
    ]

    for scenario, X_tr, y_tr, X_te, y_te in scenarios:
        logger.info("Multi-class classification — Scenario %s", scenario)
        for name, template in _multi_models():
            m = copy.deepcopy(template)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)

            row = {
                'model':       name,
                'scenario':    scenario,
                'accuracy':    accuracy_score(y_te, y_pred),
                'macro_f1':    f1_score(y_te, y_pred, average='macro', zero_division=0),
                'weighted_f1': f1_score(y_te, y_pred, average='weighted', zero_division=0),
            }
            logger.info("  %-28s [%s]  acc=%.3f  macro_f1=%.3f",
                        name, scenario, row['accuracy'], row['macro_f1'])
            results.append(row)

            _save_model(m, f"multi_{name.replace(' ', '_')}_{scenario[0]}.pkl")

    df = pd.DataFrame(results)
    save_metrics_csv(results, 'task2_multiclass_metrics.csv')
    save_bar_comparison(df, 'accuracy', 'Zadanie 2 — Accuracy (identyfikacja gazu)',
                        'Accuracy', 'task2_accuracy.png', ylim=(0, 1.1))

    best_name = (df[df['scenario'] == 'A-losowy']
                 .sort_values('accuracy', ascending=False).iloc[0]['model'])
    logger.info("Best multi-class model (Scenario A): %s", best_name)

    for scenario, X_tr, y_tr, X_te, y_te in scenarios:
        m = copy.deepcopy(dict(_multi_models())[best_name])
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        save_confusion_matrix(y_te, y_pred, GAS_LABELS,
                              f'{best_name} — Scen. {scenario}',
                              f'task2_confusion6x6_{scenario}.png')
        if scenario == 'A-losowy':
            logger.info("\nClassification report (Scenario A, %s):\n%s",
                        best_name,
                        classification_report(y_te, y_pred, target_names=GAS_LABELS, zero_division=0))

    # Feature importance for Random Forest (Scenario A)
    rf = copy.deepcopy(dict(_multi_models())['Random Forest'])
    rf.fit(X_trainA, y_trainA_0)
    _plot_feature_importance(rf, feature_names)

    return df


def _plot_feature_importance(model, feature_names: list | None, n_top: int = 20) -> None:
    """Save bar chart of top-N feature importances.

    Parameters
    ----------
    model        : fitted tree-based model with feature_importances_
    feature_names: list of feature column names (or None for indices)
    n_top        : number of top features to show
    """
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:n_top]
    labels      = ([feature_names[i] for i in top_idx] if feature_names is not None
                   else [f'feat_{i}' for i in top_idx])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(n_top), importances[top_idx], color='steelblue', alpha=0.85)
    ax.set_xticks(range(n_top))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_title(f'Top {n_top} najwazniejszych cech — Random Forest (Zadanie 2, Scen. A)', fontsize=12)
    ax.set_ylabel('Feature Importance (mean decrease impurity)')
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, 'task2_feature_importance.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Feature importance saved: %s", path)
    logger.info("Top 5 features: %s", labels[:5])


def _save_model(model, filename: str) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, filename))


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    from src.data_loader import load_data, get_feature_cols
    from src.preprocessing import random_split, batch_split

    df = load_data()
    feat_cols = get_feature_cols(df)

    (X_tA, X_eA, yg_tA, yg_eA, ya_tA, ya_eA, yc_tA, yc_eA, sA) = random_split(df)
    (X_tB, X_eB, yg_tB, yg_eB, ya_tB, ya_eB, yc_tB, yc_eB, sB) = batch_split(df)

    df_bin = run_binary_classification(X_tA, ya_tA, X_eA, ya_eA, X_tB, ya_tB, X_eB, ya_eB)
    tune_binary_rf(X_tA, ya_tA)
    df_mc  = run_multiclass_classification(X_tA, yg_tA, X_eA, yg_eA,
                                            X_tB, yg_tB, X_eB, yg_eB,
                                            feature_names=feat_cols)
    print(df_bin.to_string())
    print(df_mc.to_string())
