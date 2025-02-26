import os
import cv2
import numpy as np
from tqdm import tqdm
from joblib import dump
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    max_num_hands=2
)

DATA_DIR = "./dataTesting"

data = []
labels = []
previous_landmarks = {}  # Almacena landmarks del frame anterior por video

# Obtener nombres de clases y asignarles un índice
class_names = sorted(os.listdir(DATA_DIR))
class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
skipped_files_count = {class_name: 0 for class_name in class_names}

# Función para extraer landmarks de una imagen o frame


def extract_landmarks(image, label):
    """ Extrae landmarks de una imagen y los agrega al dataset """
    # Convert image to RGB to MediaPipe
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image with MediaPipe to detect landmarks
    result = hands_model.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data_aux = []
            motion_aux = [0] * 63
            for lm in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[lm].x
                y = hand_landmarks.landmark[lm].y
                z = hand_landmarks.landmark[lm].z
                data_aux.append(x)
                data_aux.append(y)
                data_aux.append(z)

            # Guardar datos y etiquetas
            data.append(data_aux + motion_aux)
            labels.append(label)
    else:
        return False  # Indica que no se encontraron manos
    return True  # Indica éxito

# Recorrer todas las clases
for dir_ in tqdm(os.listdir(DATA_DIR), desc="Procesando clases"):
    class_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(class_path):
        continue

    for file_name in tqdm(os.listdir(class_path), desc=f"Procesando archivos en {dir_}", leave=False):
        print(file_name)
        file_path = os.path.join(class_path, file_name)
        label = class_dict[dir_]

        if file_name.lower().endswith(('.mp4', '.avi', '.mov')):  # Procesar videos
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Advertencia: No se pudo abrir {file_path}. Saltando...")
                skipped_files_count[dir_] += 1
                print(skipped_files_count)
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # No más frames

                extract_landmarks(frame, label)
            cap.release()

        elif file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Procesar imágenes
            image = cv2.imread(file_path)
            if image is None:
                print(f"Advertencia: No se pudo abrir {file_path}. Saltando...")
                skipped_files_count[dir_] += 1
                print(skipped_files_count)
                continue
            extract_landmarks(image, label)

# Convertir a arrays de NumPy y guardar dataset
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
print(data)
print(labels)
dump({"data": data, "labels": labels}, "datamovinglettersset.joblib")

# Resumen de archivos no procesados
print("\nResumen de archivos no procesados:")
for class_name, count in skipped_files_count.items():
    print(f"{class_name}: {count} archivos saltados")

print(f"\nDataset guardado con {len(data)} muestras.")
