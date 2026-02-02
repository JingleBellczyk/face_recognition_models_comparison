# validate.py

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import sys
sys.path.append(os.path.abspath('../trening'))

import config
from models.embedding_net import EmbeddingNet
from transforms import face_transform

import logging
logging.basicConfig(level=logging.INFO)

# =============================
# KONFIGURACJA
# =============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALID_DIR = config.VAL_DIR
MODEL_PATH = config.BEST_MODEL_PATH
RESULTS_DIR = './validation_results_longer'
os.makedirs(RESULTS_DIR, exist_ok=True)

# logging =============================
# konfiguracja logów: zarówno do konsoli, jak i do pliku
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# konsola
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# plik
fh = logging.FileHandler(os.path.join(RESULTS_DIR, 'validation.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

logging.info('Logging started')
# =============================

logging.info(f'Device: {DEVICE}')
logging.info(f'Validation directory: {VALID_DIR}')
logging.info(f'Model path: {MODEL_PATH}')
logging.info(f'Results will be saved to: {RESULTS_DIR}')

# =============================
# DATASET WALIDACYJNY
# =============================
class FacePairsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.persons = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.pairs = []

        for person in self.persons:
            imgs = os.listdir(os.path.join(data_dir, person))
            if len(imgs) < 2:
                continue
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    self.pairs.append((os.path.join(data_dir, person, imgs[i]),
                                       os.path.join(data_dir, person, imgs[j]), 1))
            for other_person in [p for p in self.persons if p != person]:
                other_img = np.random.choice(os.listdir(os.path.join(data_dir, other_person)))
                self.pairs.append((os.path.join(data_dir, person, imgs[0]),
                                   os.path.join(data_dir, other_person, other_img), 0))
        logging.info(f'Generated {len(self.pairs)} pairs for validation.')

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
# WGRANIE MODELU
# =============================
model = EmbeddingNet(embedding_size=config.EMBEDDING_SIZE,
                     normalize=config.NORMALIZE_EMBEDDING).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
logging.info('Model loaded and set to evaluation mode.')

# =============================
# DATALOADER WALIDACYJNY
# =============================
val_dataset = FacePairsDataset(VALID_DIR, transform=face_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
logging.info(f'Validation DataLoader created with {len(val_loader)} batches.')

# =============================
# OBLICZANIE EMBEDDINGÓW I ODLEGŁOŚCI
# =============================
distances = []
labels = []
with torch.no_grad():
    for batch_idx, (img1, img2, label) in enumerate(val_loader, start=1):
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
        emb1 = model(img1)
        emb2 = model(img2)
        dist = (emb1 - emb2).pow(2).sum(1).sqrt().cpu().numpy()
        distances.extend(dist)
        labels.extend(label.numpy())
        if batch_idx % 10 == 0:
            logging.info(f'Processed {batch_idx}/{len(val_loader)} batches.')

# =============================
# ZAPISANIE SUROWYCH WYNIKÓW
# =============================
distances = np.array(distances)
labels = np.array(labels)
np.savez(os.path.join(RESULTS_DIR, 'validation_pairs.npz'), distances=distances, labels=labels)
logging.info('Saved raw distances and labels to validation_pairs.npz')

# =============================
# ROC Curve + AUC
# =============================
fpr, tpr, thresholds = roc_curve(labels, -distances)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300)
plt.close()
logging.info('Saved ROC curve to roc_curve.png')

# =============================
# Histogram distances
# =============================
plt.figure(figsize=(6,6))
plt.hist(distances[labels==1], bins=30, alpha=0.7, label='Positive')
plt.hist(distances[labels==0], bins=30, alpha=0.7, label='Negative')
plt.xlabel('L2 distance')
plt.ylabel('Number of pairs')
plt.title('Histogram of pairwise distances')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'histogram_distances.png'), dpi=300)
plt.close()
logging.info('Saved histogram of distances to histogram_distances.png')

# =============================
# Accuracy vs threshold
# =============================
thresholds_acc = np.linspace(0, np.max(distances), 100)
accs = []
for thr in thresholds_acc:
    preds = distances < thr
    acc = (preds == labels).mean()
    accs.append(acc)

plt.figure(figsize=(6,6))
plt.plot(thresholds_acc, accs)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Threshold')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_vs_threshold.png'), dpi=300)
plt.close()
logging.info('Saved accuracy vs threshold plot to accuracy_vs_threshold.png')
