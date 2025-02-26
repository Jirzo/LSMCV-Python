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

# Obtener nombres de clases y asignarles un índice
class_names = sorted(os.listdir(DATA_DIR))
class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
skipped_files_count = {class_name: 0 for class_name in class_names}

for class_name in tqdm(class_names, desc="Proccesing classes"):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for file_name in tqdm(os.listdir(class_path), desc=f"Proccesing files from {class_name}", leave=False):
        file_path = os.path.join(class_path, file_name)

        # Determinar si es imagen o video
        if file_name.lower().endswith(('.png', '.jpg', 'jpeg')):
            frame = cv2.imread(file_path)
            frames = [frame] # Convertir en lista para el procesarlo en videos
        elif file_name.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(file_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        else:
            continue

        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_model.process(frame_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    data_aux = []
                    for lm in hand_landmarks.landmark:
                        data_aux.extend([lm.x, lm.y, lm.z])
                        data.append(data_aux)
                        labels.append(class_dict[class_name])

# Convertir a NumPy y guardar
if data:
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    ooutput_file = "datasetMovingTesting.joblib"
    dump({"data": data, "labels": labels}, ooutput_file)
    print(f"\nDataset guardado en {ooutput_file}. Total muestras: {len(data)}")
else:
    print("No se encontraron datos válidos para el dataset.")