import os
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

SOURCE_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/archive4/val"
# SOURCE_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/train"
# TARGET_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/processed"
TARGET_DIR = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/val_cropped"

os.makedirs(TARGET_DIR, exist_ok=True)

# Insight Face initialization

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 → CPU,

# process one image
def process_image(src_path, dst_path):
    img = cv2.imread(src_path)
    if img is None:
        print(f"[ERROR] Can't read {src_path}")
        return False

    faces = app.get(img)
    if len(faces) == 0:
        print(f"[WARNING] No face: {src_path}")
        return False

    # get biggest face from detected
    face = max(faces, key=lambda x: x.bbox[2]*x.bbox[3])
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

    # crop and resize 112x112
    face_crop = img[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (112, 112))

    # save
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, face_resized)
    return True

# processing all dataset
def process_dataset(input_dir, output_dir):
    persons = os.listdir(input_dir)
    for person in tqdm(persons, desc="Przetwarzanie folderów"):
        person_src = os.path.join(input_dir, person)
        if not os.path.isdir(person_src):
            continue

        person_dst = os.path.join(output_dir, person)
        os.makedirs(person_dst, exist_ok=True)

        for img_name in os.listdir(person_src):
            src = os.path.join(person_src, img_name)
            dst = os.path.join(person_dst, img_name)
            process_image(src, dst)

# start processing all dataset
process_dataset(SOURCE_DIR, TARGET_DIR)
