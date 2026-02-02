import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

# ===== dodaj ścieżkę do treningu =====
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../trening"))
sys.path.append(ROOT)

import config
from models.embedding_net import EmbeddingNet
from transforms import face_transform

# =============================
# KONFIGURACJA
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_DIR = config.VAL_DIR
MODEL_PATH = config.BEST_MODEL_PATH
RESULTS_DIR = "./group_validation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PAIRS_PER_PERSON = 40
GROUPS = range(6)

# =============================
# MAPOWANIE ID → NAZWA GRUPY
# =============================
GROUP_ID_TO_NAME = {
    0: "Woman_Asian",
    1: "Woman_Black",
    2: "Woman_White",
    3: "Man_Asian",
    4: "Man_Black",
    5: "Man_White",
}

# =============================
# WCZYTANIE CSV (Identity → Group)
# =============================
groups_df = pd.read_csv(
    "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/identity_groups.csv"
)

groups_df["Identity"] = (
    groups_df["Identity"]
    .astype(str)
    .str.replace(r"[^0-9]", "", regex=True)
    .astype(int)
)

id_to_group = dict(zip(groups_df.Identity, groups_df.Group))

# =============================
# LICZBA OSÓB W GRUPACH (VAL_DIR)
# =============================
group_identities = defaultdict(set)

for d in os.listdir(VAL_DIR):
    if not d.startswith("n"):
        continue
    pid = int(d[1:])
    if pid in id_to_group:
        g = id_to_group[pid]
        group_identities[g].add(pid)

print("\n[INFO] Liczba osób w grupach (VAL):")
for g in GROUPS:
    group_name = GROUP_ID_TO_NAME[g]
    print(f"  {group_name}: {len(group_identities[g])} identities")

# =============================
# DATASET PER-GROUP
# =============================
class GroupFacePairsDataset(Dataset):
    def __init__(self, data_dir, target_group, transform=None):
        self.transform = transform
        self.pairs = []

        persons = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]

        # tylko osoby z danej grupy
        persons = [
            p for p in persons
            if int(p[1:]) in id_to_group
            and id_to_group[int(p[1:])] == target_group
        ]

        for person in persons:
            person_path = os.path.join(data_dir, person)
            imgs = [
                f for f in os.listdir(person_path)
                if f.lower().endswith((".jpg", ".png"))
            ]

            if len(imgs) < 2:
                continue

            # ===== POSITIVE =====
            for _ in range(min(PAIRS_PER_PERSON, len(imgs))):
                i1, i2 = random.sample(imgs, 2)
                self.pairs.append((
                    os.path.join(person_path, i1),
                    os.path.join(person_path, i2),
                    1
                ))

            # ===== NEGATIVE (ta sama grupa) =====
            others = [p for p in persons if p != person]
            if not others:
                continue

            for _ in range(min(PAIRS_PER_PERSON, len(imgs))):
                other = random.choice(others)
                other_path = os.path.join(data_dir, other)
                other_imgs = [
                    f for f in os.listdir(other_path)
                    if f.lower().endswith((".jpg", ".png"))
                ]
                if not other_imgs:
                    continue

                self.pairs.append((
                    os.path.join(person_path, random.choice(imgs)),
                    os.path.join(other_path, random.choice(other_imgs)),
                    0
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

# =============================
# MODEL
# =============================
model = EmbeddingNet(
    embedding_size=config.EMBEDDING_SIZE,
    normalize=config.NORMALIZE_EMBEDDING
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =============================
# EWALUACJA PER-GROUP
# =============================
results = []

for group in GROUPS:
    group_name = GROUP_ID_TO_NAME[group]

    if len(group_identities[group]) < 2:
        print(f"[SKIP] {group_name}: insufficient identities")
        continue

    dataset = GroupFacePairsDataset(
        VAL_DIR, group, transform=face_transform
    )

    if len(dataset) == 0:
        print(f"[SKIP] {group_name}: no pairs generated")
        continue

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    distances, labels = [], []

    with torch.no_grad():
        for img1, img2, lbl in loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            emb1, emb2 = model(img1), model(img2)
            dist = torch.norm(emb1 - emb2, dim=1).cpu().numpy()
            distances.extend(dist)
            labels.extend(lbl.numpy())

    distances = np.array(distances)
    labels = np.array(labels)

    fpr, tpr, _ = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)

    tpr_at_far = tpr[np.where(fpr <= 0.01)[0][-1]]

    results.append({
        "Group": group_name,
        "AUC": roc_auc,
        "TPR@FAR=1%": tpr_at_far,
        "NumPairs": len(labels)
    })

    # ===== ROC =====
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC – {group_name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/roc_{group_name}.png")
    plt.close()

    # ===== HISTOGRAM =====
    plt.figure()
    plt.hist(distances[labels == 1], bins=40, alpha=0.6, label="Positive")
    plt.hist(distances[labels == 0], bins=40, alpha=0.6, label="Negative")
    plt.title(f"Distances – {group_name}")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/hist_{group_name}.png")
    plt.close()

# =============================
# ZAPIS WYNIKÓW
# =============================
results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULTS_DIR}/group_results.csv", index=False)

print("\n=== WYNIKI PER-GROUP ===")
print(results_df)
