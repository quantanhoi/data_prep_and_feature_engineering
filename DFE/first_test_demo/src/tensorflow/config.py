"""
All knobs for data, model and training.
Changing a value here automatically propagates through the rest of the code.
"""

from pathlib import Path

# --------------------------------------------------------------------- data
HORSE_DATA_DIR = Path("./horse_images/data")
TRUCK_DATA_DIR = Path("./truck_images/data")

BATCH_SIZE   : int   = 64
IMAGE_SIZE   : tuple = (180, 180)
SEED         : int   = 123

# --------------------------------------------------------------------- model
NUM_FILTERS  : tuple = (32, 64, 128)   # width of the three Conv blocks
KERNEL_SIZE  : int   = 3
DENSE_UNITS  : int   = 128
DROPOUT      : float = 0.3

# ------------------------------------------------------------------- training
EPOCHS         : int   = 30
EARLY_STOP     : int   = 5
LEARNING_RATE  : float = 1e-3
