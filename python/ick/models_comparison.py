import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from PIL import Image
import random
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- KONFIGURACJA ŚCIEŻEK ---
BASE_PATH = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/ick/models/"
DATA_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/processed/"
NUM_PAIRS = 5000  # Liczba par na model (500 pos, 500 neg)

# --- 1. MODELE ---

class ResNet18Embedder(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)
    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)

def load_resnet():
    model = ResNet18Embedder(128)
    model.load_state_dict(torch.load(os.path.join(BASE_PATH, "best_face_embedding_model.pth"), map_location='cpu'))
    model.eval()
    return model

def load_tflite(name):
    interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_PATH, name))
    interpreter.allocate_tensors()
    return interpreter

# --- 2. PRZETWARZANIE (Z PADDINGIEM) ---

def preprocess_face(img_path, target_size, padding_factor=0.30):
    """
    Symuluje poprawioną logikę z Kotlina:
    1. Oblicza kwadratowy obszar twarzy.
    2. Dodaje margines (padding).
    3. Skaluje do docelowego rozmiaru.
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Zakładamy, że zdjęcia w Twoich folderach to aktualne wycinki z detektora.
    # Obliczamy środek i bazowy rozmiar (rawSize z Twojego kodu w Kotlinie)
    center_x, center_y = w / 2, h / 2
    raw_size = max(w, h)

    # Dodajemy margines (np. 1.30f w Kotlinie to padding_factor 0.30)
    padded_size = raw_size * (1 + padding_factor)

    # Wyznaczamy współrzędne wycięcia
    left = max(0, center_x - padded_size / 2)
    top = max(0, center_y - padded_size / 2)
    right = min(w, center_x + padded_size / 2)
    bottom = min(h, center_y + padded_size / 2)

    # Wycinamy i skalujemy
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped.resize((target_size, target_size), Image.BILINEAR)

# def get_resnet_embedding(model, img_path):
#     img = preprocess_face(img_path, 112)
#     img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
#     start = time.time()
#     with torch.no_grad():
#         emb = model(img_t).numpy()[0]
#    return emb, (time.time() - start) * 1000

def get_resnet_embedding(model, img_path, target_size=112, norm_type="div255"):
    img = preprocess_face(img_path, target_size)
    # ResNet zawsze div255
    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    start = time.time()
    with torch.no_grad():
        emb = model(img_t).numpy()[0]
    return emb, (time.time() - start) * 1000

# def get_tflite_embedding(interpreter, img_path):
#     input_details = interpreter.get_input_details()
#     target_size = input_details[0]['shape'][1] # 112 lub 160

#     img = preprocess_face(img_path, target_size)
#     img_array = (np.array(img).astype(np.float32) - 127.5) / 128.0
#     input_data = np.expand_dims(img_array, axis=0)

#     start = time.time()
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()
#     emb = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
#     return emb, (time.time() - start) * 1000

def get_tflite_embedding(interpreter, img_path, target_size, norm_type):
    img = preprocess_face(img_path, target_size)
    img_array = np.array(img).astype(np.float32)

    if norm_type == "minus127":
        input_data = (img_array - 127.5) / 128.0
    else:
        input_data = img_array / 255.0

    input_data = np.expand_dims(input_data, axis=0)
    input_details = interpreter.get_input_details()

    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    emb = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]

    # Obowiązkowa normalizacja L2 dla TFLite
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb, (time.time() - start) * 1000

# --- 4. GŁÓWNA PĘTLA TESTOWA ---

def run_comparison():
    print("Ładowanie modeli...")
    # Dodajemy parametry size i norm bezpośrednio do tupli
    models_info = [
        ("ResNet-18 (Mine)", load_resnet(), get_resnet_embedding, 112, "div255"),
        ("GhostFaceNet", load_tflite("ghostfacenet_optimized.tflite"), get_tflite_embedding, 112, "div255"),
        ("MobileFaceNet", load_tflite("MobileFaceNet.tflite"), get_tflite_embedding, 160, "minus127")
    ]

    folders = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    # Generowanie stałego zestawu par dla sprawiedliwego testu
    print(f"Przygotowanie {NUM_PAIRS} par testowych...")
    test_pairs = []
    for _ in range(NUM_PAIRS // 2):
        f = random.choice(folders); imgs = os.listdir(f)
        if len(imgs) >= 2:
            i1, i2 = random.sample(imgs, 2)
            test_pairs.append((os.path.join(f, i1), os.path.join(f, i2), 1))
        f1, f2 = random.sample(folders, 2)
        test_pairs.append((os.path.join(f1, os.listdir(f1)[0]), os.path.join(f2, os.listdir(f2)[0]), 0))

    plt.figure(figsize=(10, 8))
    print("\nRozpoczynam testy...")

    for name, model, emb_fn, size, norm in models_info:
        distances = []
        labels = []
        latencies = []

        print(f"Testowanie: {name}...")
        for img1, img2, label in test_pairs:
            # Przekazujemy wszystkie 4 argumenty
            e1, t1 = emb_fn(model, img1, size, norm)
            e2, t2 = emb_fn(model, img2, size, norm)

            dist = np.linalg.norm(e1 - e2)
            distances.append(dist)
            labels.append(label)
            latencies.extend([t1, t2])

        # Metryki
        # Skalowanie dystansu do zakresu 0-1 dla potrzeb krzywej ROC
        scores = 1.0 - (np.array(distances) / np.max(distances))
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        avg_time = np.mean(latencies)

        plt.plot(fpr, tpr, label=f'{name}\n(AUC: {auc_score:.3f}, EER: {eer:.3f}, {avg_time:.1f}ms)')
        print(f"DONE: {name:15} | AUC: {auc_score:.3f} | EER: {eer:.3f} | Time: {avg_time:5.1f}ms")
        plot_threshold_distribution(labels, distances, name)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Porównanie modeli Face Recognition (ROC/AUC/EER)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.2)
    plt.savefig("porownanie_modeli_results.png")
    plt.show()

def plot_threshold_distribution(labels, distances, model_name):
    plt.figure(figsize=(10, 6))

    # Rozdzielamy dystanse
    pos_dist = [d for d, l in zip(distances, labels) if l == 1]
    neg_dist = [d for d, l in zip(distances, labels) if l == 0]

    # Rysujemy histogramy zamiast sns.kdeplot
    # density=True sprawia, że wykres pokazuje gęstość (prawdopodobieństwo) zamiast surowej liczby par
    plt.hist(pos_dist, bins=50, alpha=0.5, color='g', label="Ta sama osoba (Genuine)", density=True)
    plt.hist(neg_dist, bins=50, alpha=0.5, color='r', label="Inna osoba (Impostor)", density=True)

    # Automatyczne obliczanie najlepszego progu (EER) zamiast wpisywania na sztywno 1.1
    # Używamy -distances, bo roc_curve domyślnie zakłada, że większa wartość = pozytyw
    fpr, tpr, thresholds = roc_curve(labels, -np.array(distances))
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer_threshold = -thresholds[idx] # Wracamy do dodatniego dystansu

    # Rysujemy linię sugerowanego progu
    plt.axvline(x=eer_threshold, color='k', linestyle='--',
                label=f'Optymalny próg (EER): {eer_threshold:.3f}')

    plt.title(f'Rozkład dystansów L2 - {model_name}')
    plt.xlabel('Dystans Euklidesowy (L2)')
    plt.ylabel('Gęstość (Prawdopodobieństwo)')
    plt.legend()
    plt.grid(alpha=0.3)

    # Zapis do pliku, żebyś mogła go łatwo obejrzeć
    plt.savefig(f"dist_{model_name.replace(' ', '_')}.png")
    plt.show()

    print(f"Model: {model_name} -> Sugerowany próg do Kotlina: {eer_threshold:.4f}")

def run_lfw_benchmark(lfw_path, min_images=10):
    print(f"Przygotowuję benchmark LFW z: {lfw_path}")

    # 1. Filtrowanie folderów - bierzemy tylko te, które mają >= min_images
    all_people = [d for d in os.listdir(lfw_path) if os.path.isdir(os.path.join(lfw_path, d))]
    qualified_people = []

    for person in all_people:
        person_dir = os.path.join(lfw_path, person)
        if len(os.listdir(person_dir)) >= min_images:
            qualified_people.append(person)

    print(f"Znaleziono {len(qualified_people)} osób spełniających kryterium (min. {min_images} zdjęć).")

    if len(qualified_people) < 2:
        print("BŁĄD: Za mało osób do przeprowadzenia testu negatywnego!")
        return

    # 2. Generowanie par testowych (np. 1000 par)
    test_pairs = []
    num_bench_pairs = 1000

    for _ in range(num_bench_pairs // 2):
        # Para pozytywna (ta sama osoba)
        p = random.choice(qualified_people)
        p_path = os.path.join(lfw_path, p)
        imgs = os.listdir(p_path)
        i1, i2 = random.sample(imgs, 2)
        test_pairs.append((os.path.join(p_path, i1), os.path.join(p_path, i2), 1))

        # Para negatywna (różne osoby)
        p1, p2 = random.sample(qualified_people, 2)
        i1 = random.choice(os.listdir(os.path.join(lfw_path, p1)))
        i2 = random.choice(os.listdir(os.path.join(lfw_path, p2)))
        test_pairs.append((os.path.join(lfw_path, p1, i1), os.path.join(lfw_path, p2, i2), 0))

    # 3. Parametry modeli do porównania
    configs = [
        {"name": "GhostFaceNet", "size": 112, "norm": "div255", "model": load_tflite("ghostfacenet_optimized.tflite")},
        {"name": "MobileFaceNet", "size": 160, "norm": "minus127", "model": load_tflite("MobileFaceNet.tflite")},
        {"name": "ResNet-18 (Ours)", "size": 112, "norm": "div255", "model": load_resnet()}
    ]

    # 4. Pętla testowa
    for cfg in configs:
        distances = []
        labels = []
        print(f"\n--- Benchmark LFW dla: {cfg['name']} ---")

        for img1, img2, label in test_pairs:
            # Pamiętaj, aby Twoje funkcje emb_fn przyjmowały (model, path, size, norm)
            if "ResNet" in cfg['name']:
                e1, _ = get_resnet_embedding(cfg['model'], img1, cfg['size'], cfg['norm'])
                e2, _ = get_resnet_embedding(cfg['model'], img2, cfg['size'], cfg['norm'])
            else:
                e1, _ = get_tflite_embedding(cfg['model'], img1, cfg['size'], cfg['norm'])
                e2, _ = get_tflite_embedding(cfg['model'], img2, cfg['size'], cfg['norm'])

            distances.append(np.linalg.norm(e1 - e2))
            labels.append(label)

        # Wywołujemy wykres rozkładu (tę funkcję z plt.hist, którą robiliśmy wcześniej)
        plot_threshold_distribution(labels, distances, f"LFW_{cfg['name']}")

# Wywołanie:

if __name__ == "__main__":
    # run_comparison()
    run_lfw_benchmark("/home/agata-ner/Documents/semestr9/glebokie_uczenie/test_modeli/lfw-deepfunneled", min_images=10)
