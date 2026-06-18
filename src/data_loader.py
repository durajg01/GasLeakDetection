"""Load and cache the UCI Gas Sensor Array Drift dataset.

Primary source: local batch1.dat … batch10.dat files in DATA_DIR.
Fallback: UCI ML Repository download (id=270) when .dat files are absent.

Run standalone:  python -m src.data_loader
"""
import logging
import os

import numpy as np
import pandas as pd

from config import (
    ALARM_THRESHOLDS, CACHE_FILE,
    DATA_DIR, GAS_NAMES, UCI_DATASET_ID,
)

logger = logging.getLogger(__name__)

_N_FEATURES = 128
_FEAT_COLS  = [f'feat_{i}' for i in range(1, _N_FEATURES + 1)]


# ─── public API ───────────────────────────────────────────────────────────────

def load_data(use_cache: bool = True) -> pd.DataFrame:
    """Return the full gas sensor dataset as a DataFrame.

    Columns: feat_1 … feat_128, batch_id, gas_class,
             concentration_ppmv, gas_name, alarm.

    Parameters
    ----------
    use_cache : bool
        Load from parquet cache if it exists.

    Returns
    -------
    pd.DataFrame  shape (13910, 133)
    """
    if use_cache and os.path.exists(CACHE_FILE):
        logger.info("Loading data from cache: %s", CACHE_FILE)
        return pd.read_parquet(CACHE_FILE)

    dat_files = [os.path.join(DATA_DIR, f'batch{i}.dat') for i in range(1, 11)]
    if all(os.path.exists(p) for p in dat_files):
        logger.info("Loading data from local .dat files in %s", DATA_DIR)
        df = _load_from_dat_files(dat_files)
    else:
        logger.info("Local .dat files not found — downloading from UCI (id=%d)", UCI_DATASET_ID)
        df = _load_from_uci()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)
    logger.info("Data cached to %s", CACHE_FILE)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature column names (everything except metadata)."""
    meta = {'batch_id', 'gas_class', 'concentration_ppmv', 'gas_name', 'alarm'}
    return [c for c in df.columns if c not in meta]


# ─── local .dat loader ────────────────────────────────────────────────────────

def _load_from_dat_files(dat_files: list) -> pd.DataFrame:
    """Parse all batch*.dat files into a single DataFrame.

    File format (LIBSVM-style):
        gas_class;concentration_ppmv  1:val  2:val  …  128:val
    """
    records = []
    for batch_id, filepath in enumerate(dat_files, start=1):
        with open(filepath, 'r') as fh:
            for line in fh:
                rec = _parse_dat_line(line, batch_id)
                if rec is not None:
                    records.append(rec)
        logger.info("  batch %2d — %d samples read", batch_id,
                    sum(1 for r in records if r['batch_id'] == batch_id))

    df = pd.DataFrame(records, columns=_FEAT_COLS + ['batch_id', 'gas_class', 'concentration_ppmv'])
    df = _add_derived_cols(df)

    logger.info("Dataset shape: %s  |  Missing values: %d", df.shape, df.isnull().sum().sum())
    logger.info("Gas distribution:   %s", df['gas_name'].value_counts().to_dict())
    logger.info("Alarm distribution: %s", df['alarm'].value_counts().to_dict())
    return df


def _parse_dat_line(line: str, batch_id: int) -> dict | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    gas_str, conc_str = parts[0].split(';')
    gas_class    = int(gas_str)
    conc_ppmv    = float(conc_str)
    feat_vals    = {int(k): float(v) for token in parts[1:] for k, v in [token.split(':')]}
    record       = {f'feat_{i}': feat_vals.get(i, 0.0) for i in range(1, _N_FEATURES + 1)}
    record.update({'batch_id': batch_id, 'gas_class': gas_class, 'concentration_ppmv': conc_ppmv})
    return record


# ─── UCI fallback loader ──────────────────────────────────────────────────────

def _load_from_uci() -> pd.DataFrame:
    """Download dataset from UCI ML Repository and return as DataFrame."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    X_raw   = dataset.data.features
    y_raw   = dataset.data.targets
    logger.info("Features: %s  |  Targets: %s", X_raw.shape, y_raw.shape)

    # Force Arrow/object string columns to float64
    df = X_raw.apply(lambda s: pd.to_numeric(s.astype(str), errors='coerce'))
    df = df.fillna(df.mean()).fillna(0.0)
    df.columns = _FEAT_COLS  # rename to feat_1 … feat_128

    target_cols = y_raw.columns.tolist()
    gas_col  = _find_col(target_cols, ['class', 'gas', 'target', 'label']) or target_cols[0]
    conc_col = _find_col(target_cols, ['conc', 'ppm', 'concentration'])
    if conc_col is None and len(target_cols) > 1:
        conc_col = target_cols[1]
    logger.info("Using target columns: gas='%s'  conc='%s'", gas_col, conc_col)

    raw_vals = y_raw[gas_col].astype(str)
    if raw_vals.str.contains(':').any():
        split     = raw_vals.str.split(':', expand=True)
        gas_class = split[0].str.strip().astype(int)
        conc_ppmv = split[1].str.strip().astype(float)
    else:
        gas_class = raw_vals.str.strip().astype(int)
        conc_ppmv = (pd.Series(y_raw[conc_col].values.astype(float), index=raw_vals.index)
                     if conc_col else pd.Series(np.zeros(len(raw_vals)), index=raw_vals.index))

    # Normalise to 1-indexed if the repo returned 0-indexed classes
    unique_cls = sorted(gas_class.unique().tolist())
    if unique_cls and min(unique_cls) == 0 and max(unique_cls) <= 5:
        logger.info("Shifting 0-indexed classes → 1-indexed")
        gas_class = gas_class + 1

    # Assign batch IDs using known sizes
    from config import BATCH_SIZES
    assert sum(BATCH_SIZES) == len(df), \
        f"BATCH_SIZES sum {sum(BATCH_SIZES)} ≠ dataset rows {len(df)}"
    batch_ids: list[int] = []
    for b, size in enumerate(BATCH_SIZES, 1):
        batch_ids.extend([b] * size)

    meta = pd.DataFrame({
        'batch_id':           batch_ids,
        'gas_class':          gas_class.values,
        'concentration_ppmv': conc_ppmv.values,
    }, index=df.index)
    df = pd.concat([df, meta], axis=1)
    df = _add_derived_cols(df)

    logger.info("Dataset shape: %s  |  Missing values: %d", df.shape, df.isnull().sum().sum())
    logger.info("Alarm distribution: %s", df['alarm'].value_counts().to_dict())
    return df


# ─── shared helpers ───────────────────────────────────────────────────────────

def _add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    df['gas_name'] = df['gas_class'].map(GAS_NAMES).fillna('Unknown')
    unknown = (df['gas_name'] == 'Unknown').sum()
    if unknown:
        logger.warning("Unmapped gas_class values: %s  (count=%d)",
                       df.loc[df['gas_name'] == 'Unknown', 'gas_class'].unique().tolist(),
                       unknown)
    df['alarm'] = (df['concentration_ppmv']
                   >= df['gas_class'].map(ALARM_THRESHOLDS).fillna(9999)).astype(int)
    return df


def _find_col(cols: list, keywords: list) -> str | None:
    for col in cols:
        if any(k in col.lower() for k in keywords):
            return col
    return None


# ─── standalone ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    df = load_data(use_cache=False)
    print(df.head(3))
    print(f"\nShape: {df.shape}")
    print(f"Gas distribution:\n{df['gas_name'].value_counts()}")
    print(f"Alarm distribution:\n{df['alarm'].value_counts()}")
