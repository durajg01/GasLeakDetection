"""Shared helpers: save metrics to CSV/JSON, plot confusion matrices, comparison charts."""
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config import FIGURES_DIR, METRICS_DIR

logger = logging.getLogger(__name__)


def save_metrics_csv(records: list, filename: str) -> None:
    """Append-or-create a CSV of metric dicts in METRICS_DIR.

    Parameters
    ----------
    records : list of dict
    filename : str  e.g. 'task1_binary_metrics.csv'
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, filename)
    pd.DataFrame(records).to_csv(path, index=False)
    logger.info("Metrics saved: %s", path)


def save_metrics_json(data: dict, filename: str) -> None:
    """Save a metrics dict as JSON in METRICS_DIR.

    Parameters
    ----------
    data : dict
    filename : str  e.g. 'tuning_results.json'
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved: %s", path)


def save_confusion_matrix(
    y_true, y_pred, labels: list, title: str, filename: str
) -> None:
    """Plot and save a confusion matrix to FIGURES_DIR.

    Parameters
    ----------
    y_true   : array-like of true labels
    y_pred   : array-like of predicted labels
    labels   : list of display label strings
    title    : figure title
    filename : output file name (saved under FIGURES_DIR)
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)))
    # Pass explicit labels so the matrix is always n×n even when some classes
    # are absent from y_true/y_pred (e.g. Scenario B batch split)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax,
        colorbar=False,
        cmap='Blues',
        xticks_rotation='vertical' if n > 4 else 'horizontal',
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    logger.info("Confusion matrix saved: %s", path)


def save_bar_comparison(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    ylim: tuple | None = None,
) -> None:
    """Grouped bar chart comparing metric across models and scenarios.

    Parameters
    ----------
    df       : DataFrame with columns 'model', 'scenario', and the metric column
    metric   : column name to plot
    title    : figure title
    ylabel   : y-axis label
    filename : output file name (saved under FIGURES_DIR)
    ylim     : optional (ymin, ymax) tuple
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pivot = df.pivot(index='model', columns='scenario', values=metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind='bar', ax=ax, width=0.6, colormap='Paired', edgecolor='gray')
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis='x', rotation=20)
    ax.legend(title='Scenariusz')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    logger.info("Comparison chart saved: %s", path)
