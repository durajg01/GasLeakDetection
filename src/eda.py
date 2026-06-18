"""Exploratory Data Analysis — generates all figures to results/figures/.

Run standalone:  python -m src.eda
"""
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import ALARM_THRESHOLDS, FIGURES_DIR, GAS_NAMES, RANDOM_STATE
from src.data_loader import get_feature_cols

logger = logging.getLogger(__name__)

COLORS = sns.color_palette('tab10', 6)
sns.set_theme(style='whitegrid', palette='tab10')


def run_eda(df: pd.DataFrame) -> None:
    """Run the full EDA pipeline and save all figures.

    Parameters
    ----------
    df : full dataset DataFrame from load_data()
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    feature_cols = get_feature_cols(df)

    logger.info("EDA: gas distribution")
    plot_gas_distribution(df)

    logger.info("EDA: concentration distributions")
    plot_concentration_distribution(df)

    logger.info("EDA: sensor profiles")
    plot_sensor_profiles(df, feature_cols)

    logger.info("EDA: PCA")
    plot_pca(df, feature_cols)

    logger.info("EDA: t-SNE")
    plot_tsne(df, feature_cols)

    logger.info("EDA: correlation matrix")
    plot_correlation_matrix(df, feature_cols)

    logger.info("EDA: sensor drift (batch 1 vs 10)")
    plot_sensor_drift(df, feature_cols)

    logger.info("EDA complete — figures saved to %s", FIGURES_DIR)


def plot_gas_distribution(df: pd.DataFrame) -> None:
    """Bar chart of gas class counts overall and stacked per batch.

    Parameters
    ----------
    df : dataset DataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    gas_counts = df['gas_name'].value_counts()
    axes[0].bar(gas_counts.index, gas_counts.values, color=COLORS)
    axes[0].set_title('Rozklad klas gazow (caly dataset)', fontsize=13)
    axes[0].set_xlabel('Gaz')
    axes[0].set_ylabel('Liczba probek')
    for i, v in enumerate(gas_counts.values):
        axes[0].text(i, v + 30, str(v), ha='center', fontsize=9)

    pivot = df.groupby(['batch_id', 'gas_name']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_title('Rozklad gazow per batch (nierownosc!)', fontsize=13)
    axes[1].set_xlabel('Batch ID')
    axes[1].set_ylabel('Liczba probek')
    axes[1].legend(title='Gaz', bbox_to_anchor=(1.01, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    _save(fig, 'eda_gas_distribution.png')


def plot_concentration_distribution(df: pd.DataFrame) -> None:
    """Histograms of concentration per gas with alarm threshold lines.

    Parameters
    ----------
    df : dataset DataFrame
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for i, (gas_id, gas_name) in enumerate(GAS_NAMES.items()):
        data = df[df['gas_class'] == gas_id]['concentration_ppmv']
        thr  = ALARM_THRESHOLDS[gas_id]
        axes[i].hist(data, bins=40, color=COLORS[i], alpha=0.75, edgecolor='white')
        axes[i].axvline(thr, color='red', linestyle='--', linewidth=2, label=f'Prog: {thr} ppmv')
        axes[i].set_title(f'{gas_name} (klasa {gas_id})', fontsize=12)
        axes[i].set_xlabel('Stezenie (ppmv)')
        axes[i].set_ylabel('Liczba probek')
        axes[i].legend(fontsize=8)
        pct = (data >= thr).mean() * 100
        axes[i].text(0.97, 0.85, f'Alarm: {pct:.1f}%', transform=axes[i].transAxes,
                     ha='right', fontsize=9, color='red')

    plt.suptitle('Rozklad stezenia per gaz z progiem alarmowym', fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, 'eda_concentration.png')


def plot_sensor_profiles(df: pd.DataFrame, feature_cols: list) -> None:
    """Line chart of mean feature values for first 16 features (one per sensor).

    Parameters
    ----------
    df           : dataset DataFrame
    feature_cols : ordered list of feature column names
    """
    sample_features  = feature_cols[:16]
    mean_profiles    = df.groupby('gas_name')[sample_features].mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    for j, gas_name in enumerate(mean_profiles.index):
        ax.plot(range(16), mean_profiles.loc[gas_name].values,
                marker='o', markersize=4, label=gas_name)

    ax.set_title('Srednie profile pierwszych 16 cech (po jednej z kazdego czujnika)', fontsize=12)
    ax.set_xlabel('Indeks czujnika (0-15)')
    ax.set_ylabel('Srednia wartosc cechy')
    ax.legend(title='Gaz')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    _save(fig, 'eda_profiles.png')


def plot_pca(df: pd.DataFrame, feature_cols: list) -> None:
    """PCA scatter plots coloured by gas class and alarm label.

    Parameters
    ----------
    df           : dataset DataFrame
    feature_cols : list of feature column names
    """
    X_all    = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X_all)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pca   = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, (gas_id, gas_name) in enumerate(GAS_NAMES.items()):
        mask = df['gas_class'].values == gas_id
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=[COLORS[i]], s=5, alpha=0.5, label=gas_name)
    axes[0].set_title(
        f'PCA (2 skladowe) — po klasie gazu\n'
        f'PC1={pca.explained_variance_ratio_[0]:.1%}  '
        f'PC2={pca.explained_variance_ratio_[1]:.1%}',
        fontsize=10,
    )
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(markerscale=3, title='Gaz')

    for alarm_val, label, color in [(0, 'Brak alarmu', 'steelblue'), (1, 'ALARM', 'crimson')]:
        mask = df['alarm'].values == alarm_val
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, s=5, alpha=0.4, label=label)
    axes[1].set_title('PCA — po etykiecie alarmowej', fontsize=10)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].legend(markerscale=3)

    plt.tight_layout()
    _save(fig, 'eda_pca.png')
    logger.info("PCA explained variance: %.1f%%", pca.explained_variance_ratio_.sum() * 100)


def plot_tsne(df: pd.DataFrame, feature_cols: list, n_samples: int = 3000) -> None:
    """t-SNE visualisation on a random subsample coloured by gas class.

    Parameters
    ----------
    df           : dataset DataFrame
    feature_cols : list of feature column names
    n_samples    : number of random samples to use (full t-SNE is slow)
    """
    X_all    = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X_all)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng(RANDOM_STATE)
    idx    = rng.choice(len(df), n_samples, replace=False)
    labels = df['gas_class'].values[idx]

    logger.info("Running t-SNE on %d samples (may take a minute)...", n_samples)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=40, max_iter=500)
    X_2d = tsne.fit_transform(X_scaled[idx])

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (gas_id, gas_name) in enumerate(GAS_NAMES.items()):
        mask = labels == gas_id
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[COLORS[i]], s=8, alpha=0.6, label=gas_name)
    ax.set_title(f't-SNE (probka {n_samples}) — po klasie gazu', fontsize=12)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend(markerscale=3, title='Gaz')
    plt.tight_layout()
    _save(fig, 'eda_tsne.png')


def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list, n_features: int = 32) -> None:
    """Lower-triangle correlation heatmap for the first n_features features.

    Parameters
    ----------
    df           : dataset DataFrame
    feature_cols : list of feature column names
    n_features   : how many features to include (default 32)
    """
    corr = df[feature_cols[:n_features]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, cbar_kws={'shrink': 0.8}, linewidths=0.3)
    ax.set_title(f'Macierz korelacji — pierwsze {n_features} cech', fontsize=12)
    plt.tight_layout()
    _save(fig, 'eda_correlation.png')

    high_corr = (corr.abs() > 0.9) & (corr != 1.0)
    logger.info("Pairs with |corr| > 0.9: %d", high_corr.sum().sum() // 2)


def plot_sensor_drift(df: pd.DataFrame, feature_cols: list, gas_id: int = 1) -> None:
    """Bar chart comparing mean DR features between batch 1 and batch 10.

    Parameters
    ----------
    df           : dataset DataFrame
    feature_cols : list of feature column names
    gas_id       : which gas class to compare (default 1 = Ethanol)
    """
    from config import GAS_NAMES as _GN
    dr_cols  = feature_cols[0::8][:16]   # first feature (DR) of each of 16 sensors
    gas_name = _GN[gas_id]

    b1  = df[(df['batch_id'] == 1)  & (df['gas_class'] == gas_id)][dr_cols].mean()
    b10 = df[(df['batch_id'] == 10) & (df['gas_class'] == gas_id)][dr_cols].mean()

    x, w = np.arange(len(dr_cols)), 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w / 2, b1.values,  w, label='Batch 1 (sty 2008)',  color='steelblue', alpha=0.85)
    ax.bar(x + w / 2, b10.values, w, label='Batch 10 (lut 2011)', color='coral',     alpha=0.85)
    ax.set_title(f'Dryft sensorow: srednie DR dla {gas_name} (batch 1 vs batch 10)', fontsize=12)
    ax.set_xlabel('Czujnik')
    ax.set_ylabel('Srednia wartosc DR')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(len(dr_cols))])
    ax.legend()
    plt.tight_layout()
    _save(fig, 'eda_drift_sensor.png')

    drift_pct = ((b10 - b1).abs() / b1.abs().clip(lower=1e-9) * 100).mean()
    logger.info("Mean relative DR change batch 1→10: %.1f%%", drift_pct)


def _save(fig, filename: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved %s", path)


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    from src.data_loader import load_data
    run_eda(load_data())
