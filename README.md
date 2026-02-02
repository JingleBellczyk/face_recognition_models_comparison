W folderze python:
- service/serwer.py - implementacja serwisu, który zwraca embedding dla modelu resnet-18
- trening - zawiera kod użyty do treningu modelu
- /validation i /ick - użyte do walidacji
- /cutting_images - użyty do rozpoznawania twarzy i wycinania zdjęć w trakcie treningu i walidacji
- kotlin_preparing - kod użyty do konwersji modelu pytorch (.pth) do onnx

W folderze kotlin:
- znajduje sie implementacja frontendu
- w /assets znajdują się modele w formacie .tflite do rozpoznawania twarzy

Zdjęcia ze zbioru vgg-dataset: https://www.kaggle.com/datasets/hearfool/vggface2
