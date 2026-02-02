# convert_to_tflite.py

import torch
import sys
import os
sys.path.append(os.path.abspath('../trening'))  # ścieżka do EmbeddingNet i config
from models.embedding_net import EmbeddingNet
import config

# =============================
# KONFIGURACJA
# =============================
DEVICE = torch.device('cpu')  # TFLite wymaga CPU
MODEL_PATH = config.BEST_MODEL_PATH
TS_PATH = 'face_embedding_model.pt'   # pośredni TorchScript
TFLITE_PATH = 'face_embedding_model.tflite'

# =============================
# ŁADUJ MODEL
# =============================
model = EmbeddingNet(
    embedding_size=config.EMBEDDING_SIZE,
    normalize=config.NORMALIZE_EMBEDDING
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded and set to evaluation mode.")

# =============================
# Dummy input
# =============================
dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

# =============================
# Export TorchScript
# =============================
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save(TS_PATH)
print(f"TorchScript model saved to {TS_PATH}")

# =============================
# Konwersja do TFLite
# =============================
try:
    import torch_tflite
except ImportError:
    raise ImportError(
        "Nie znaleziono torch_tflite. Zainstaluj: pip install torch-tflite"
    )

tflite_model = torch_tflite.export(
    traced_model,
    input_shape=(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE),
    dtype=torch.float32
)

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {TFLITE_PATH}")

# =============================
# Wyświetl rozmiary wejścia/wyjścia
# =============================
print(f"Input size: {dummy_input.shape}")
with torch.no_grad():
    out = model(dummy_input)
print(f"Output size: {out.shape}")
