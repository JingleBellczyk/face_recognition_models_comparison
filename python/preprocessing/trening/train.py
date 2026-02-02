# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

import config
from datasets.triplet_dataset import TripletFaceDataset
from models.embedding_net import EmbeddingNet
from transforms import face_transform
import logging
logging.basicConfig(filename='train.log', level=logging.INFO)

# =============================
# USTAWIENIE SEED (reprodukowalność)
# =============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.SEED)

# =============================
# DEVICE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# DATASET & DATALOADER
# =============================
dataset = TripletFaceDataset(
    config.TRAIN_DIR,
    transform=face_transform
)

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)

# =============================
# MODEL
# =============================
model = EmbeddingNet(
    embedding_size=config.EMBEDDING_SIZE,
    normalize=config.NORMALIZE_EMBEDDING
).to(device)

# =============================
# LOSS & OPTIMIZER
# =============================
criterion = nn.TripletMarginLoss(margin=config.MARGIN)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)

# =============================
# TRAINING
# =============================
best_loss = float("inf")

for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0.0

    print(f"\n[INFO] Rozpoczynam epokę {epoch+1}/{config.EPOCHS}")

    for batch_idx, (anchor, positive, negative) in enumerate(loader, start=1):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        # ===== forward =====
        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        # ===== loss =====
        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ===== logowanie batchy co N batchy =====
        if batch_idx % config.LOG_EVERY_N_BATCHES == 0 or batch_idx == len(loader):
            # 🔹 dodatkowe logi: średnie odległości między embeddingami
            with torch.no_grad():
                pos_dist = (emb_a - emb_p).pow(2).sum(1).sqrt().mean().item()
                neg_dist = (emb_a - emb_n).pow(2).sum(1).sqrt().mean().item()

            print(
                f"[Batch {batch_idx}/{len(loader)}] "
                f"loss: {loss.item():.4f} | "
                f"pos_dist: {pos_dist:.4f} | "
                f"neg_dist: {neg_dist:.4f} | "
                f"emb_a min/max: {emb_a.min().item():.4f}/{emb_a.max().item():.4f} | "
                f"emb_p min/max: {emb_p.min().item():.4f}/{emb_p.max().item():.4f} | "
                f"emb_n min/max: {emb_n.min().item():.4f}/{emb_n.max().item():.4f}"
            )

    avg_loss = total_loss / len(loader)
    print(f"[INFO] Epoka {epoch+1} zakończona. Średnia strata: {avg_loss:.4f}")

    # ===== zapis najlepszego modelu =====
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), config.BEST_MODEL_PATH)
        print(f"[INFO] Nowy najlepszy model zapisany! avg_loss={avg_loss:.4f}")
        logging.info(f"Nowy najlepszy model zapisany! Epoka {epoch+1}, avg_loss={avg_loss:.4f}")

    # ===== dodatkowo zapis modelu co epokę =====
    torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch+1}.pth")
    logging.info(f"Model zapisany po epoce {epoch+1}, avg_loss={avg_loss:.4f}")


    # ===== scheduler krok (zmiana LR po epokach) =====
    scheduler.step()
    print(f"[INFO] LR po epoce: {scheduler.get_last_lr()[0]:.6f}")
