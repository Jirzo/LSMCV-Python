# Comentarios clave:
# 1 -Procesamiento de directorios y clases:
#       Cada subdirectorio dentro de DATA_DIR se considera una clase.
#       Las imágenes dentro de cada subdirectorio representan ejemplos de esa clase.
# 2- Extracción de landmarks:
#       MediaPipe devuelve coordenadas normalizadas (valores entre 0 y 1) para los landmarks detectados en la mano.
#       Estas coordenadas se almacenan como pares x, y en la lista data_aux.
# 3- Etiquetas:
#       dir_ se utiliza como la etiqueta para cada conjunto de landmarks procesados. Esto ayuda a conectar los datos con su categoría correspondiente.
# 4- Serialización con pickle:
#       Los datos y las etiquetas se guardan en un archivo binario (data.pickle) para su uso posterior, como entrenamiento de modelos de machine learning.

import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()
DATA_DIR = "./data"
# Inicializar MediaPipe para la detección de manos.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


data = []  # Lista para almacenar las coordenadas procesadas de los landmarks.
label = []  # Lista para almacenar las etiquetas correspondientes.

# Recorrer cada directorio en la carpeta de datos.
# Cada directorio representa una clase diferente (por ejemplo, un gesto o una categoría).
for dir_ in os.listdir(DATA_DIR):
    # Procesar cada imagen dentro del directorio actual.
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Lista temporal para almacenar los landmarks de una sola imagen.

        # Leer la imagen para convertirla de BGR (formato OpenCV) a RGB (formato MediaPipe).
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe para detectar las manos y sus landmarks.
        result = hands.process(img_rgb)

        # Si se detectan landmarks en la imagen:
        if result.multi_hand_landmarks:
            # Iterar a través de todas las manos detectadas (en caso de múltiples manos).
            for hand_landmarks in result.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            # Agregar los landmarks procesados a la lista principal de datos.
            data.append(data_aux)

            # Agregar la etiqueta (clase) correspondiente a la lista de etiquetas.
            # 'dir_' representa la clase basada en el nombre del directorio.
            label.append(dir_)

# Guardar los datos y etiquetas en un archivo binario utilizando pickle.
# Esto permite que los datos sean reutilizados más tarde para entrenamiento o evaluación del modelo.
f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": label}, f)
f.close()
