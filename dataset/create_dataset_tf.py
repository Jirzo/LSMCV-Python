import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from settings.collect_image import DATA_DIR

N_FRAMES = 30  # Número de frames por secuencia

def datasetTFCreation_LSTM(hands_model):
    class_names = sorted(os.listdir(DATA_DIR))
    class_dict = {name: idx for idx, name in enumerate(class_names)}
    skipped_images_count = {class_name: 0 for class_name in class_names}

    sequences = []
    labels = []

    for class_name in tqdm(class_names, desc="Procesando clases"):
        class_path = os.path.join(DATA_DIR, class_name)

        if not os.path.isdir(class_path):
            continue

        sequence = []
        label_idx = class_dict[class_name]

        for img_path in tqdm(sorted(os.listdir(class_path)), desc=f"Imágenes en {class_name}", leave=False):
            img_full_path = os.path.join(class_path, img_path)

            img = cv2.imread(img_full_path)
            if img is None:
                skipped_images_count[class_name] += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands_model.process(img_rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]  # tomamos solo una mano
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
                    data_aux.append(landmark.z)

                sequence.append(data_aux)

                if len(sequence) == N_FRAMES:
                    sequences.append(sequence.copy())
                    labels.append(label_idx)
                    sequence = []  # Reiniciar para siguiente secuencia
            else:
                skipped_images_count[class_name] += 1

    print("\nResumen de imágenes saltadas por clase:")
    for class_name, count in skipped_images_count.items():
        print(f"{class_name}: {count} imágenes saltadas")

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    save_to_tfrecord(sequences, labels, 'landmark_sequences.tfrecord')

        # Guardar metadata
    metadata = {
        "num_classes": len(class_dict),
        "class_dict": class_dict
    }
    with open("landmarks_metadata.json", "w") as f:
        json.dump(metadata, f)


def save_to_tfrecord(sequences, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in tqdm(range(len(sequences)), desc=f"Guardando en {output_file}"):
            feature = {
                'landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=sequences[i].flatten())),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f"\n¡Datos guardados exitosamente en {output_file}!")
