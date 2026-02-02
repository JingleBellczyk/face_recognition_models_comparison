
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRENING_DIR = os.path.join(BASE_DIR, "trening")

sys.path.append(TRENING_DIR)
import config
from datasets.triplet_dataset import TripletFaceDataset
from transforms import face_transform


dataset = TripletFaceDataset(
    data_dir=config.TRAIN_DIR,
    transform=face_transform
)

# Pobranie jednej próbki (indeks nie ma znaczenia – losowanie on-the-fly)
anchor, positive, negative = dataset[0]

print("Anchor:", anchor.shape)
print("Positive:", positive.shape)
print("Negative:", negative.shape)


def show_triplet(anchor, positive, negative):
    # tensor → numpy (C,H,W → H,W,C)
    def tensor_to_img(t):
        img = t.permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5) + 0.5  # denormalizacja
        return np.clip(img, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].imshow(tensor_to_img(anchor))
    axs[0].set_title("Anchor")
    axs[0].axis("off")

    axs[1].imshow(tensor_to_img(positive))
    axs[1].set_title("Positive")
    axs[1].axis("off")

    axs[2].imshow(tensor_to_img(negative))
    axs[2].set_title("Negative")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


# Wyświetlenie przykładowej trójki
show_triplet(anchor, positive, negative)
