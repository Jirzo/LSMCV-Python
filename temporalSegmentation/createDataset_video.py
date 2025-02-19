import os
import cv2
import numpy as np
from tqdm import tqdm
from joblib import dump
import mediapipe as mp

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,  # Reducido para detectar más manos en videos de baja calidad
    min_tracking_confidence=0.3,
    max_num_hands=2
)

DATA_VIDEO_DIR = "./data"

data = []
labels = []
previous_landmarks = {}  # Diccionario para almacenar landmarks del frame anterior por video

class_names = sorted(os.listdir(DATA_VIDEO_DIR))
class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
skipped_videos_count = {class_name: 0 for class_name in class_names}


for dir_ in tqdm(os.listdir(DATA_VIDEO_DIR), desc="Procesando clases"):
    class_path = os.path.join(DATA_VIDEO_DIR, dir_)

    # Verificar si es un directorio
    if not os.path.isdir(class_path):
        continue

    for video_name in tqdm(os.listdir(class_path), desc=f"Procesando videos de {dir_}", leave=False):
        video_path = os.path.join(class_path, video_name)

        # Cargar el video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Advertencia: No se pudo abrir el video {video_path}. Saltando...")
            skipped_videos_count[dir_] += 1
            continue

        frame_count = 0
        previous_landmarks[video_name] = None  # Inicializar landmarks anteriores para este video

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Salir cuando no haya más frames

            # Convertir frame a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar frame con MediaPipe
            result = hands_model.process(frame_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    data_aux = []
                    motion_aux = [] # Lista para los vectores de movimiento

                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x)
                        data_aux.append(lm.y)
                        data_aux.append(lm.z)
                        prev_lm = None
                        if previous_landmarks[video_name] is not None:
                            # Necesitamos el indice i para acceder al landmark previo correcto
                            # Usaremos enumerate para obtener el indice
                            for i, prev_lm_actual in enumerate(previous_landmarks[video_name].landmar):
                                # Comparamos indices para asegurarnos de usar el landmark previocorrecto
                                if hand_landmarks.landmark.index(lm) == i:
                                    prev_lm == prev_lm_actual
                                    break

                            motion_x = lm.x - prev_lm.x
                            motion_y = lm.y - prev_lm.y
                            motion_z = lm.z - prev_lm.z
                            motion_aux.append(motion_x)
                            motion_aux.append(motion_y)
                            motion_aux.append(motion_z)
                        else:
                           # Para el primer frame, añade 0s como vectores de movimiento
                            motion_aux.extend([0, 0, 0])  # Vectores cero para el primer frame

                    # Guardar landmarks y etiqueta de la clase
                    data.append(data_aux + motion_aux)
                    labels.append(class_dict[dir_])

            frame_count += 1

        cap.release()

        if frame_count == 0:
            print(f"Advertencia: {video_path} no contenía frames utilizables.")

# Convertir datos a arrays de NumPy
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Guardar el dataset con joblib
output_file = "datamovinglettersset.joblib"
dump({"data": data, "labels": labels}, output_file)

# Resumen de videos saltados
print("\nResumen de videos saltados por clase:")
for class_name, count in skipped_videos_count.items():
    print(f"{class_name}: {count} videos saltados")

print(f"\nDataset guardado en {output_file}. Total muestras: {len(data)}")
