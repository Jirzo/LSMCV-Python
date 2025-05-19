import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from settings.collect_image import DATA_DIR

N_FRAMES = 1  # Número de frames por secuencia

def datasetTFCreation(hands_model):
    class_names = sorted(os.listdir(DATA_DIR))
    class_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

    sequences = []
    labels = []
    skipped_sequences = {class_name: 0 for class_name in class_names}

    for dir_ in tqdm(os.listdir(DATA_DIR), desc="Procesando clases"):
        class_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(class_path):
            continue

        image_paths = sorted(os.listdir(class_path))
        num_sequences = len(image_paths) // N_FRAMES

        for seq_idx in tqdm(range(num_sequences), desc=f"Procesando secuencias en {dir_}", leave=False):
            sequence = []
            valid_sequence = True
            for frame_idx in range(N_FRAMES):
                img_idx = seq_idx * N_FRAMES + frame_idx
                img_full_path = os.path.join(class_path, image_paths[img_idx])
                img = cv2.imread(img_full_path)
                if img is None:
                    valid_sequence = False
                    break

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands_model.process(img_rgb)

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]  # solo la primera mano
                    frame_data = []
                    for landmark in hand_landmarks.landmark:
                        frame_data.append(landmark.x)
                        frame_data.append(landmark.y)
                    sequence.append(frame_data)
                else:
                    valid_sequence = False
                    break

            if valid_sequence and len(sequence) == N_FRAMES:
                sequences.append(sequence)
                labels.append(class_dict[dir_])
            else:
                skipped_sequences[dir_] += 1

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print("\nResumen de secuencias saltadas:")
    for class_name, count in skipped_sequences.items():
        print(f"{class_name}: {count} secuencias inválidas")

    save_to_tfrecord(sequences, labels, "sequences_dataset.tfrecord")


def save_to_tfrecord(sequences, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in tqdm(range(len(sequences)), desc=f"Escribiendo {output_file}"):
            feature = {
                'landmarks': tf.train.Feature(float_list=tf.train.FloatList(value=sequences[i].flatten())),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f"\n✅ Dataset guardado en {output_file}")
