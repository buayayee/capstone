"""
config.py
---------
All training hyperparameters and paths in one place.
Edit this file to tune your model.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_CSV = BASE_DIR / "data" / "labeled" / "dataset_augmented.csv"   # use augmented dataset
MODEL_OUTPUT_DIR = BASE_DIR / "models" / "distilbert-fraud-classifier"
LOGS_DIR = BASE_DIR / "models" / "logs"

# ── Model ──────────────────────────────────────────────────────────────────
PRETRAINED_MODEL = "distilbert-base-uncased"  # small, fast, accurate enough
NUM_LABELS = 2  # 0=genuine, 1=fraudulent

# ── Tokenization ───────────────────────────────────────────────────────────
MAX_TOKEN_LENGTH = 512  # DistilBERT max sequence length

# ── Training ───────────────────────────────────────────────────────────────
TEST_SIZE = 0.2           # 80% train, 20% test
RANDOM_SEED = 42
NUM_EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1        # % of training steps used for LR warmup

# ── Class Imbalance ────────────────────────────────────────────────────────
# Set to True if fraudulent samples are still fewer than genuine after augmentation
USE_CLASS_WEIGHTS = True
