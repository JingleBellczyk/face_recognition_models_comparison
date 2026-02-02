# config.py
import torch

# =============================
# ŚCIEŻKI
# =============================
TRAIN_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/processed"
VAL_DIR   = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/val_cropped"

CHECKPOINT_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/trening/checkpoints"
BEST_MODEL_PATH = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/trening/checkpoints/best_face_embedding_model.pth"

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4

MARGIN = 0.5  # Triplet Loss margin
EMBEDDING_SIZE = 128

# =============================
# =============================
DATASET_LENGTH = 5000   # sztuczna długość epoki (on-the-fly triplets)
NUM_WORKERS = 4
PIN_MEMORY = True

# =============================
# OBRAZY
# =============================
IMAGE_SIZE = 112
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

# =============================
# MODEL
# =============================
BACKBONE = "resnet18"
PRETRAINED = True
NORMALIZE_EMBEDDING = True

# =============================
# LOSOWOŚĆ
# =============================
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOGOWANIE
LOG_EVERY_N_BATCHES = 10