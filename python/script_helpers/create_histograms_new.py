import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Ścieżki
# =====================
CSV_PATH = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/identity_groups.csv"
TRAIN_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/processed"
VAL_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/val_cropped"

# =====================
# MAPOWANIE GRUP
# =====================
GROUP_ID_TO_NAME = {
    0: "Woman_Asian",
    1: "Woman_Black",
    2: "Woman_White",
    3: "Man_Asian",
    4: "Man_Black",
    5: "Man_White",
}

# =====================
# Wczytaj CSV
# =====================
groups_df = pd.read_csv(CSV_PATH)
groups_df["Identity"] = groups_df["Identity"].astype(int)

# =====================
# Funkcja: ID z folderów
# =====================
def get_identities_from_dir(path):
    identities = []
    for name in os.listdir(path):
        if name.startswith("n"):
            try:
                identities.append(int(name[1:]))  # n000123 -> 123
            except ValueError:
                pass
    return identities

train_ids = get_identities_from_dir(TRAIN_DIR)
val_ids = get_identities_from_dir(VAL_DIR)

# =====================
# Filtruj tylko dostępne ID
# =====================
train_groups = groups_df[groups_df["Identity"].isin(train_ids)]
val_groups = groups_df[groups_df["Identity"].isin(val_ids)]

# =====================
# Liczebność grup
# =====================
train_counts = train_groups["Group"].value_counts().sort_index()
val_counts = val_groups["Group"].value_counts().sort_index()

# =====================
# Przygotuj etykiety osi X
# =====================
group_ids = sorted(GROUP_ID_TO_NAME.keys())
group_labels = [GROUP_ID_TO_NAME[g] for g in group_ids]

train_values = [train_counts.get(g, 0) for g in group_ids]
val_values = [val_counts.get(g, 0) for g in group_ids]

# =====================
# Rysowanie
# =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].bar(group_labels, train_values)
axes[0].set_title("Zbiór treningowy")
axes[0].set_xlabel("Grupa")
axes[0].set_ylabel("Liczba osób")
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(group_labels, val_values)
axes[1].set_title("Zbiór walidacyjny")
axes[1].set_xlabel("Grupa")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()
