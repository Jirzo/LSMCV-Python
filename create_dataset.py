import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm  # Para mostrar el progreso en terminal
from joblib import dump

# Directorio que contiene los datos organizados por clases
DATA_DIR = "./data"

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Listas para almacenar los datos y etiquetas procesadas
data = []  # Coordenadas de landmarks
labels = []  # Etiquetas de clase

# Diccionario para mapear clases a índices
class_names = sorted(os.listdir(DATA_DIR))  # Asumimos que los directorios son las clases
class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

# Diccionario para contar imágenes saltadas por cada clase
skipped_images_count = {class_name: 0 for class_name in class_names}

# Recorrer cada directorio/clase en la carpeta de datos
for dir_ in tqdm(os.listdir(DATA_DIR), desc="Procesando clases"):
    class_path = os.path.join(DATA_DIR, dir_)

    # Verificar si el directorio es válido
    if not os.path.isdir(class_path):
        continue

    # Procesar cada imagen dentro del directorio actual
    for img_path in tqdm(os.listdir(class_path), desc=f"Procesando imágenes de {dir_}", leave=False):
        img_full_path = os.path.join(class_path, img_path)

        # Leer y validar la imagen
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Advertencia: No se pudo cargar la imagen {img_full_path}. Saltando...")
            skipped_images_count[dir_] += 1
            continue

        # Convertir la imagen a RGB para MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe para detectar landmarks
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:  # Si se detectan manos
            for hand_landmarks in result.multi_hand_landmarks:
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

                # Agregar landmarks y etiquetas a las listas principales
                data.append(data_aux)
                labels.append(class_dict[dir_])  # Usar índice numérico de la clase
        else:
            print(f"Advertencia: No se detectaron landmarks en {img_full_path}. Saltando...")
            skipped_images_count[dir_] += 1

# Convertir a arrays de NumPy
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Guardar el dataset como archivo usando joblib
output_file = "data_tensorflow.joblib"
dump({"data": data, "labels": labels}, output_file)

# Mostrar resumen de imágenes saltadas
print("\nResumen de imágenes saltadas por clase:")
for class_name, count in skipped_images_count.items():
    print(f"{class_name}: {count} imágenes saltadas")

print(f"\nDataset procesado y guardado en {output_file}.")

