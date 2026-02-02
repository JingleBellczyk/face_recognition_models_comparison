import torch
import tensorflow as tf
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

# --- DEFINICJA KLASY (Musi być identyczna jak przy trenowaniu) ---
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(EmbeddingNet, self).__init__()
        # Twoja architektura: ResNet18 z podmienionym FC
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        # Normalizacja L2 jest kluczowa dla dystansu euklidesowego
        return F.normalize(x, p=2, dim=1)

# --- 1. Sprawdzenie Twojego modelu PyTorch (.pth) ---
def check_pth(model_path):
    print(f"\n--- Model PyTorch: {os.path.basename(model_path)} ---")
    if not os.path.exists(model_path):
        print(f"❌ Błąd: Nie znaleziono pliku pod ścieżką: {model_path}")
        return

    try:
        device = torch.device("cpu")
        # Zakładamy standardowe 128, jeśli wyjście będzie inne, PyTorch wyrzuci błąd przy load_state_dict
        model = EmbeddingNet(embedding_size=128)

        # Wczytanie wag
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Dummy input: (Batch, Channels, Height, Width)
        dummy_input = torch.randn(1, 3, 112, 112)

        with torch.no_grad():
            output = model(dummy_input)

        print(f"✅ Model wczytany poprawnie.")
        print(f"Wejście testowe: {dummy_input.shape}")
        print(f"Wyjście (Embedding size): {output.shape[1]}")
    except Exception as e:
        print(f"❌ Błąd podczas sprawdzania .pth: {e}")

# --- 2. Sprawdzenie modeli TFLite ---
def check_tflite(model_path):
    print(f"\n--- Model TFLite: {os.path.basename(model_path)} ---")
    if not os.path.exists(model_path):
        print(f"❌ Błąd: Nie znaleziono pliku: {model_path}")
        return

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Wejście (Shape): {input_details[0]['shape']}")
        print(f"Wyjście (Shape): {output_details[0]['shape']}")
        print(f"Typ danych wyjściowych: {output_details[0]['dtype']}")
    except Exception as e:
        print(f"❌ Błąd podczas sprawdzania .tflite: {e}")



# --- URUCHOMIENIE ---
# Popraw ścieżki na poprawne w Twoim systemie (usuń '/' na początku jeśli to ścieżki względne)
BASE_PATH = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/ick/models/"

try:
    # Modele TFLite
    check_tflite(os.path.join(BASE_PATH, "ghostfacenet_optimized.tflite"))
    check_tflite(os.path.join(BASE_PATH, "MobileFaceNet.tflite"))

    # Twój model PyTorch
    check_pth(os.path.join(BASE_PATH, "best_face_embedding_model.pth"))
except Exception as e:
    print(f"Błąd główny: {e}")


