import os
import cv2
import time
import numpy as np
import mediapipe as mp

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


# Configurar el modelo de detección de manos
hands_model = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Directorios de almacenamiento
DATA_VIDEO_DIR = "./video"
# Letras del alfabeto permitidas
alphabet = list("ABCDEFGHIJKL'LL'MNÑOPQRSTUVWXYZ")
alphabetC = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "LL": 12,  # "LL" como una letra separada
    "M": 13,
    "N": 14,
    "Ñ": 15,
    "O": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "U": 22,
    "V": 23,
    "W": 24,
    "X": 25,
    "Y": 26,
    "Z": 27
}

# Inicializar la captura de video (0 para webcam, 1 si es una segunda cámara)
cap = cv2.VideoCapture(1)

# Leer el primer frame y convertirlo a escala de grises
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (5, 5), 0)

# Configurar el códec y variables de grabación
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Códec MP4
video_index = 0
video_writer = None
frames_detected = 0
recording_start_time = None
recording_duration = 5  # Duracion del clip

# Diccionario para almacenar landmarks previos de cada video
previous_landmarks = {}

while True:
    print("\nSelecciona la letra a capturar (A-Z). Ingresa '1' para salir.")
    selected_letter = input("Letra: ").strip().upper()

    if selected_letter == "1":
        break
    if selected_letter not in alphabetC:
        print("Entrada no válida. Solo letras A-Z.")
        continue

    # Crear directorio de almacenamiento si no existe
    class_dir = os.path.join(DATA_VIDEO_DIR, selected_letter)
    os.makedirs(class_dir, exist_ok=True)

    print(f'\nPreparando captura para la letra "{selected_letter}"...')
    print('Presiona "1" para detener la grabación.')
    counter = 0
    while counter < 15:
        ret, frame2 = cap.read()
        if not ret:
            break

        # Convertir a RGB para MediaPipe
        img_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        result = hands_model.process(img_rgb)

        # Convertir a escala de grises para detección de movimiento
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.GaussianBlur(frame2_gray, (5, 5), 0)

        # Detectar de movimiento
        diff = cv2.absdiff(frame1_gray, frame2_gray)
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        print("Thresh su,: ", np.sum(thresh))

        # Si se detecta una mano y hay suficiente movimiento, iniciar grabación
        if result.multi_hand_landmarks and np.sum(thresh) > 10000:
            if video_writer is None:
                video_filename = os.path.join(
                    class_dir, f"movimiento_{video_index}.mp4")
                video_writer = cv2.VideoWriter(
                    video_filename, fourcc, 120.0, (frame2.shape[1], frame2.shape[0]))
                print(f"🔴 Grabando {video_filename}")
                recording_start_time = time.time()  # Inicia el timer

                # Inicializamos landmarks previos paar este video
                previous_landmarks[video_filename] = None

            if recording_start_time is not None and time.time() - recording_start_time <= recording_duration:
                video_writer.write(frame2)
                frames_detected += 1

                # Visualizacion del timer en el frame
                time_left = recording_duration - \
                    (time.time() - recording_start_time)
                cv2.putText(frame2, f"Grabando... {time_left:.1}s", (
                    10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Visualizacion de vectores de mmovimiento
                if result.multi_hand_landmarks and previous_landmarks[video_filename] is not None:
                    for i, lm in enumerate(hand_landmarks.landmarks):
                        prev_lm = previous_landmarks[video_filename].landmarks[i]
                        x1 = int(lm.x * frame2.shape[1])
                        y1 = int(lm.y * frame2.shape[0])
                        x2 = int(prev_lm.x * frame2.shape[1])
                        y2 = int(prev_lm.y * frame2.shape[0])
                        # tipLength para mejor visualización
                        cv2.arrowedLine(frame2, (x2, y2), (x1, y1),
                                        (0, 0, 255), 2, tipLength=0.2)

        else:
            if video_writer is not None:
                video_writer.release()
                print(f"✅ Video guardado ({frames_detected} frames)")
                counter += 1
                video_writer = None
                frames_detected = 0
                video_index += 1
                recording_start_time = None

                # Reiniciar landmarks previos para este video
                previous_landmarks[video_filename] = None

        frame1_gray = frame2_gray.copy()

        # Dibujar landmarks en la imagen
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame2,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Mostrar ventana con detección de movimiento y video
        cv2.imshow("Video", frame2)
        cv2.imshow("Movimiento Detectado", thresh)
        if cv2.waitKey(25) & 0xFF == ord("1"):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
if video_writer is not None:
    video_writer.release()
