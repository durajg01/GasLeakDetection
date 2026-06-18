"""Project-wide constants: alarm thresholds, paths, random seed."""
import os

RANDOM_STATE = 42

GAS_NAMES = {
    1: 'Ethanol',
    2: 'Ethylene',
    3: 'Ammonia',
    4: 'Acetaldehyde',
    5: 'Acetone',
    6: 'Toluene',
}

# Alarm thresholds in ppmv, based on OSHA/NIOSH occupational exposure guidelines.
# Conservative early-warning levels — below legal limits to allow timely evacuation.
# Ethanol  (1): OSHA TWA ~1000 ppm; 200 ppm chosen as early warning
# Ethylene (2): LEL ~27 000 ppm; 100 ppm as early detection
# Ammonia  (3): NIOSH IDLH = 300 ppm; 250 ppm threshold
# Acetald. (4): NIOSH TWA = 200 ppm; 150 ppm threshold
# Acetone  (5): OSHA TWA = 1000 ppm; 300 ppm conservative threshold
# Toluene  (6): OSHA PEL = 200 ppm; 50 ppm (toxic at low concentrations)
ALARM_THRESHOLDS = {1: 200, 2: 100, 3: 250, 4: 150, 5: 300, 6: 50}

# Batch sizes derived from the local .dat files (Jan 2008 – Feb 2011, 10 batches)
BATCH_SIZES = [445, 1244, 1586, 161, 197, 2300, 3613, 294, 470, 3600]

UCI_DATASET_ID = 270

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data', 'raw')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')
CACHE_FILE  = os.path.join(DATA_DIR, 'gas_data.parquet')
