from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os

app = Flask(__name__)

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_size)
        self.backbone = backbone
    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)

# --- ŁADOWANIE MODELU ---
MODEL_PATH = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/trening/checkpoints/best_face_embedding_model.pth"
device = torch.device("cpu")
model = EmbeddingNet()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Wczytano wagi z {MODEL_PATH}")
else:
    print(f"⚠️ Nie znaleziono {MODEL_PATH}. Uruchamiam z losowymi wagami do testów API.")

model.eval()

# Transformacje
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Brak pliku w żądaniu"}), 400

        file = request.files['file']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            embedding = model(input_tensor)

        # Zwracamy listę 128 liczb
        return jsonify({
            "status": "success",
            "embedding": embedding.tolist()[0]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)