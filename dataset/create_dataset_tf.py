import os
import cv2 
import json 
import numpy as np # Biblioteca NumPy para operaciones numéricas, especialmente con arrays.
from tqdm import tqdm # Módulo para mostrar barras de progreso, útil para visualizar el avance de bucles.
import tensorflow as tf # Biblioteca TensorFlow para aprendizaje automático, utilizada para crear y guardar el dataset.
from settings.collect_image import DATA_DIR # Importa la ruta del directorio donde se encuentran las imágenes, definida en otro archivo.

N_FRAMES = 30  # Define el número de frames por secuencia. Cada secuencia estará compuesta por los landmarks de N_FRAMES imágenes consecutivas.

def datasetTFCreation_LSTM(hands_model):
    """
    Crea un dataset en formato TFRecord a partir de imágenes, extrayendo los landmarks de las manos
    utilizando un modelo de detección de manos (hands_model). Las secuencias de landmarks se
    agrupan y se asocian con sus respectivas etiquetas de clase.

    Args:
        hands_model: Un modelo cargado (ej. de MediaPipe Hands) capaz de procesar imágenes
                     y detectar landmarks de manos.
    """
    # Obtiene una lista ordenada de los nombres de las clases (directorios) dentro de DATA_DIR.
    class_names = sorted(os.listdir(DATA_DIR))
    # Crea un diccionario que mapea cada nombre de clase a un índice numérico único.
    class_dict = {name: idx for idx, name in enumerate(class_names)}
    # Inicializa un diccionario para contar las imágenes que no pudieron ser procesadas por cada clase.
    skipped_images_count = {class_name: 0 for class_name in class_names}

    sequences = [] # Lista para almacenar las secuencias de landmarks. Cada elemento es una secuencia de N_FRAMES.
    labels = []    # Lista para almacenar las etiquetas numéricas correspondientes a cada secuencia.

    # Itera sobre cada nombre de clase con una barra de progreso.
    for class_name in tqdm(class_names, desc="Procesando clases"):
        # Construye la ruta completa al directorio de la clase actual.
        class_path = os.path.join(DATA_DIR, class_name)

        # Si la ruta no es un directorio, salta a la siguiente iteración.
        if not os.path.isdir(class_path):
            continue

        sequence = [] # Lista temporal para construir la secuencia de landmarks para la clase actual.
        # Obtiene el índice numérico de la clase actual.
        label_idx = class_dict[class_name]

        # Itera sobre cada imagen en el directorio de la clase actual con una barra de progreso anidada.
        for img_path in tqdm(sorted(os.listdir(class_path)), desc=f"Imágenes en {class_name}", leave=False):
            # Construye la ruta completa a la imagen actual.
            img_full_path = os.path.join(class_path, img_path)

            # Lee la imagen usando OpenCV.
            img = cv2.imread(img_full_path)
            # Si la imagen no se pudo leer (ej. archivo corrupto o no es una imagen), incrementa el contador
            # de imágenes saltadas y pasa a la siguiente imagen.
            if img is None:
                skipped_images_count[class_name] += 1
                continue

            # Convierte la imagen de BGR (formato de OpenCV) a RGB (formato esperado por muchos modelos).
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Procesa la imagen con el modelo de detección de manos para obtener los resultados.
            result = hands_model.process(img_rgb)

            # Verifica si se detectaron landmarks de manos en la imagen.
            if result.multi_hand_landmarks:
                # Toma solo los landmarks de la primera mano detectada (asumiendo una sola mano por imagen).
                hand_landmarks = result.multi_hand_landmarks[0]
                data_aux = [] # Lista temporal para almacenar las coordenadas (x, y, z) de los landmarks.
                # Itera sobre cada landmark de la mano.
                for landmark in hand_landmarks.landmark:
                    # Agrega las coordenadas x, y, z del landmark a data_aux.
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
                    data_aux.append(landmark.z)

                # Agrega los landmarks de la mano actual a la secuencia temporal.
                sequence.append(data_aux)

                # Si la secuencia temporal ha alcanzado el número de frames deseado (N_FRAMES).
                if len(sequence) == N_FRAMES:
                    # Agrega una copia de la secuencia completa a la lista de secuencias finales.
                    sequences.append(sequence.copy())
                    # Agrega la etiqueta numérica correspondiente a la secuencia.
                    labels.append(label_idx)
                    # Reinicia la secuencia temporal para empezar a recolectar la siguiente secuencia.
                    sequence = []
            else:
                # Si no se detectaron landmarks de manos en la imagen, incrementa el contador de imágenes saltadas.
                skipped_images_count[class_name] += 1

    # Imprime un resumen de cuántas imágenes fueron saltadas por cada clase.
    print("\nResumen de imágenes saltadas por clase:")
    for class_name, count in skipped_images_count.items():
        print(f"{class_name}: {count} imágenes saltadas")

    # Convierte las listas de secuencias y etiquetas a arrays de NumPy con los tipos de datos especificados.
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    # Llama a la función para guardar las secuencias y etiquetas en un archivo TFRecord.
    save_to_tfrecord(sequences, labels, 'landmark_sequences.tfrecord')

    # Guarda metadatos relevantes como el número de clases y el diccionario de clases en un archivo JSON.
    metadata = {
        "num_classes": len(class_dict),
        "class_dict": class_dict
    }
    with open("landmarks_metadata.json", "w") as f:
        json.dump(metadata, f)


def save_to_tfrecord(sequences, labels, output_file):
    """
    Guarda las secuencias de landmarks y sus etiquetas correspondientes en un archivo TFRecord.

    Args:
        sequences (np.array): Array NumPy de las secuencias de landmarks.
        labels (np.array): Array NumPy de las etiquetas de las secuencias.
        output_file (str): Nombre del archivo TFRecord de salida.
    """
    # Abre un escritor de TFRecord para el archivo de salida.
    with tf.io.TFRecordWriter(output_file) as writer:
        # Itera sobre cada secuencia y su etiqueta con una barra de progreso.
        for i in tqdm(range(len(sequences)), desc=f"Guardando en {output_file}"):
            # Crea un diccionario de características para el ejemplo de TensorFlow.
            # 'landmarks': Convierte el array de landmarks a una lista plana de floats.
            # 'label': Convierte la etiqueta a una lista de enteros.
            feature = {
                'landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=sequences[i].flatten())),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))
            }
            # Crea un objeto Example de TensorFlow a partir de las características.
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serializa el ejemplo a una cadena de bytes y lo escribe en el archivo TFRecord.
            writer.write(example.SerializeToString())

    # Imprime un mensaje de éxito una vez que se han guardado todos los datos.
    print(f"\n¡Datos guardados exitosamente en {output_file}!")