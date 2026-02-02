import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# =============================
# IMPORTY Z PROJEKTU
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../trening"))

import config
from models.embedding_net import EmbeddingNet
from transforms import face_transform

# =============================
# DEVICE & MODEL
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingNet(
    embedding_size=config.EMBEDDING_SIZE,
    normalize=config.NORMALIZE_EMBEDDING
).to(DEVICE)

model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

# =============================
# POMOCNICZE FUNKCJE
# =============================
def load_image(path):
    img = Image.open(path).convert("RGB")
    return face_transform(img).unsqueeze(0).to(DEVICE)


def l2_distance(e1, e2):
    return torch.norm(e1 - e2, p=2).item()


def denormalize(img):
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * 0.5 + 0.5
    return np.clip(img, 0, 1)


def plot_pair(img1, img2, dist, title):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(img1)
    axs[0].set_title("Image A")
    axs[0].axis("off")

    axs[1].imshow(img2)
    axs[1].set_title("Image B")
    axs[1].axis("off")

    plt.suptitle(f"{title}\nL2 distance = {dist:.4f}", fontsize=11)
    plt.tight_layout()
    plt.show()

# =============================
# WYBÓR PRZYKŁADOWYCH PAR
# =============================
VAL_DIR = config.VAL_DIR
persons = [d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]

# =============================
# PARA POZYTYWNA
# =============================
person = random.choice(persons)
imgs = os.listdir(os.path.join(VAL_DIR, person))
img1_name, img2_name = random.sample(imgs, 2)

img1_path = os.path.join(VAL_DIR, person, img1_name)
img2_path = os.path.join(VAL_DIR, person, img2_name)

print("\n[POSITIVE PAIR]")
print(f"Person ID A: {person}")
print(f"Person ID B: {person}")
print(f"Image A: {img1_name}")
print(f"Image B: {img2_name}")

img1 = load_image(img1_path)
img2 = load_image(img2_path)

with torch.no_grad():
    emb1 = model(img1)
    emb2 = model(img2)

pos_dist = l2_distance(emb1, emb2)
print(f"L2 distance (positive): {pos_dist:.4f}")

plot_pair(
    denormalize(img1),
    denormalize(img2),
    pos_dist,
    title="Positive pair (same identity)"
)

# =============================
# PARA NEGATYWNA
# =============================
person_a, person_b = random.sample(persons, 2)
img_a = random.choice(os.listdir(os.path.join(VAL_DIR, person_a)))
img_b = random.choice(os.listdir(os.path.join(VAL_DIR, person_b)))

img1_path = os.path.join(VAL_DIR, person_a, img_a)
img2_path = os.path.join(VAL_DIR, person_b, img_b)

print("\n[NEGATIVE PAIR]")
print(f"Person ID A: {person_a}")
print(f"Person ID B: {person_b}")
print(f"Image A: {img_a}")
print(f"Image B: {img_b}")

img1 = load_image(img1_path)
img2 = load_image(img2_path)

with torch.no_grad():
    emb1 = model(img1)
    emb2 = model(img2)

neg_dist = l2_distance(emb1, emb2)
print(f"L2 distance (negative): {neg_dist:.4f}")

plot_pair(
    denormalize(img1),
    denormalize(img2),
    neg_dist,
    title="Negative pair (different identities)"
)
