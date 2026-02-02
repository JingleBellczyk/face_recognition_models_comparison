import os
import shutil

# Ścieżka do folderu processed (źródło nazw folderów)
processed_dir = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/processed"

# Ścieżka do folderu train (tam chcemy usuwać)
train_dir = "/home/agata-ner/Documents/semestr9/glebokie_uczenie/preprocessing/images/train"

# Pobieramy listę folderów w processed
folders_to_remove = [
    f for f in os.listdir(processed_dir)
    if os.path.isdir(os.path.join(processed_dir, f))
]

# Usuwamy foldery z train, jeśli istnieją
for folder in folders_to_remove:
    folder_path = os.path.join(train_dir, folder)
    if os.path.exists(folder_path):
        print(f"Usuwam: {folder_path}")
        shutil.rmtree(folder_path)
    else:
        print(f"Nie znaleziono folderu w train: {folder_path}")

print("Gotowe!")
