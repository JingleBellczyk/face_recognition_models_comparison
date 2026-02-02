# validate.py (POPROWIONY)

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import sys

# Dodajemy ścieżkę do folderu treningowego
sys.path.append(os.path.abspath('../trening'))

import config
from models.embedding_net import EmbeddingNet
from transforms import face_transform

import logging

# =============================
# KONFIGURACJA
# =============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALID_DIR = config.VAL_DIR
MODEL_PATH = config.BEST_MODEL_PATH
RESULTS_DIR = './validation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# logging =============================
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Usuwamy stare handlery, żeby uniknąć duplikacji logów
if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(os.path.join(RESULTS_DIR, 'validation.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

logging.info(f'Device: {DEVICE}')
logging.info(f'Validation directory: {VALID_DIR}')
logging.info(f'Normalize Embedding: {config.NORMALIZE_EMBEDDING}')

# =============================
# POPRAWIONY DATASET WALIDACYJNY
# =============================
class FacePairsDataset(Dataset):
    def __init__(self, data_dir, transform=None, pairs_per_person=50):c
        self.data_dir = data_dir
        self.transform = transform
        # Filtrujemy tylko foldery, ignorując pliki systemowe
        self.persons = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.pairs = []

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        for person in self.persons:
            person_path = os.path.join(data_dir, person)
            # Pobieramy tylko pliki graficzne
            imgs = [f for f in os.listdir(person_path) if f.lower().endswith(valid_extensions)]

            if len(imgs) < 2:
                continue

            # 1. PARY POZYTYWNE (Ta sama osoba)
            # Losujemy pary, by nie brać wszystkich możliwych kombinacji (balans)
            num_pos = min(len(imgs), pairs_per_person)
            for _ in range(num_pos):
                img1, img2 = random.sample(imgs, 2)
                self.pairs.append((os.path.join(person_path, img1),
                                   os.path.join(person_path, img2), 1))

            # 2. PARY NEGATYWNE (Różne osoby)
            # Tyle samo par negatywnych co pozytywnych dla danej osoby
            for _ in range(num_pos):
                other_person = random.choice([p for p in self.persons if p != person])
                other_person_path = os.path.join(data_dir, other_person)
                other_imgs = [f for f in os.listdir(other_person_path) if f.lower().endswith(valid_extensions)]

                if not other_imgs: continue

                img1 = random.choice(imgs)
                img2 = random.choice(other_imgs)
                self.pairs.append((os.path.join(person_path, img1),
                                   os.path.join(other_person_path, img2), 0))

        logging.info(f'Generated {len(self.pairs)} balanced pairs (Positive/Negative ratio 1:1).')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

# =============================
# WGRANIE MODELU I DANYCH
# =============================
model = EmbeddingNet(embedding_size=config.EMBEDDING_SIZE,
                     normalize=config.NORMALIZE_EMBEDDING).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

val_dataset = FacePairsDataset(VALID_DIR, transform=face_transform, pairs_per_person=60)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =============================
# OBLICZANIE DYSTANSÓW
# =============================
distances = []
labels = []

with torch.no_grad():
    for img1, img2, label in val_loader:
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        emb1, emb2 = model(img1), model(img2)

        # Dystans Euklidesowy (L2)
        dist = torch.norm(emb1 - emb2, dim=1).cpu().numpy()
        distances.extend(dist)
        labels.extend(label.numpy())

distances = np.array(distances)
labels = np.array(labels)

# =============================
# GENEROWANIE WYKRESÓW
# =============================

# 1. Histogram (Kluczowy dla Twojego pytania)
plt.figure(figsize=(8, 6))
# Używamy density=True, żeby pola pod wykresami były równe 1 (lepiej widać nachodzenie na siebie)
plt.hist(distances[labels == 1], bins=40, alpha=0.6, label='Positive (Same Person)', color='tab:blue')
plt.hist(distances[labels == 0], bins=40, alpha=0.6, label='Negative (Different Persons)', color='tab:orange')
plt.xlabel('L2 Distance')
plt.ylabel('Frequency')
plt.title('Distribution of Pairwise Distances')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'histogram_distances.png'))
plt.close()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(labels, -distances) # -distances bo mniejszy dystans = większe podobieństwo
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
plt.close()

logging.info(f'Validation complete. AUC: {roc_auc:.4f}')