"""Central configuration: paths, hyper-parameters, seeds and environment detection.

Everything that the rest of the pipeline needs to be reproducible lives here.
Works locally (CPU, smoke tests) and on Google Colab (GPU, full runs).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

IS_COLAB = is_colab()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if IS_COLAB:
    PROJECT_ROOT = Path(os.environ.get(
        "PROJECT_ROOT",
        "/content/encoder_ffn_dns"
    ))
    if not PROJECT_ROOT.exists():
        PROJECT_ROOT = Path.cwd()

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR    = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR  = PROJECT_ROOT / "models"

# Sub-directories per dataset
for _ds in ("marques", "bccc"):
    for _kind in ("eda", "training", "evaluation", "ablation"):
        (FIGURES_DIR / _kind / _ds).mkdir(parents=True, exist_ok=True)

for _d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Quick-test mode (QUICK_TEST=1 → tiny subset + few epochs)
# ---------------------------------------------------------------------------
QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
MARQUES_URL  = "https://raw.githubusercontent.com/vanilink/DNS-Poisoning-Detection/main/marques_dns_dataset.csv"
MARQUES_PATH = DATA_DIR / "marques_dns_dataset.csv"

BCCC_KAGGLE_SLUG = "bcccdatasets/malicious-dns-and-attacks-bccc-cic-bell-dns-2024"
BCCC_DIR         = DATA_DIR / "bccc_dns"

# BCCC loading caps
BCCC_MAX_BENIGN_ROWS = 600_000   # cap benign; keep ALL malicious
BCCC_CHUNK_SIZE      = 100_000

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM   = 64
NUM_HEADS       = 4
KEY_DIM         = 16       # = EMBEDDING_DIM // NUM_HEADS
FFN_HIDDEN      = 32
DROPOUT_RATE    = 0.35
DROPOUT_MID     = 0.20
LABEL_SMOOTHING = 0.05
L2_REG          = 1e-4

# BCCC override (stronger regularisation)
BCCC_DROPOUT_RATE = 0.50
BCCC_DROPOUT_MID  = 0.30
BCCC_L2_REG       = 5e-4
BCCC_LR           = 3e-4

# Tokenisation
MAX_VOCAB_MARQUES = 800
MAX_VOCAB_BCCC    = 2000
MAX_SEQ_LEN       = 16

# Training
BATCH_SIZE   = 64 if QUICK_TEST else 256
EPOCHS_MAX   = 3  if QUICK_TEST else 80
LR_INITIAL   = 1e-3
ES_PATIENCE  = 15
ES_MIN_DELTA = 1e-4

# Split
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# Deduplication (Marques)
DEDUP_MAX_OCCURRENCES = 5

TARGET_COL = "Class"

assert EMBEDDING_DIM % NUM_HEADS == 0
assert KEY_DIM == EMBEDDING_DIM // NUM_HEADS

# Apply seed at import time
set_global_seed(SEED)
