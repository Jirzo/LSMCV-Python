import tensorflow as tf
import json


with open("landmarks_metadata.json", "r") as f:
    metadata = json.load(f)

# Parámetros esperados
timesteps = 10  # Número de frames por secuencia
num_landmarks = 21
features_per_landmark = 3
num_classes = metadata["num_classes"]

# Función para parsear el TFRecord
def _parse_function(example_proto):
    feature_description = {
        'landmarks': tf.io.FixedLenFeature([timesteps * 63], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Restaurar el shape original (timesteps, 21, 3)
    landmarks = tf.reshape(parsed_example["landmarks"], (timesteps, num_landmarks, features_per_landmark))
    label = parsed_example["label"]
    return landmarks, label

# Cargar el dataset
def load_dataset(tfrecord_path, batch_size=32, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset

train_ds = load_dataset("sequences_dataset.tfrecord")

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((timesteps, 63), input_shape=(timesteps, 21, 3)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds, epochs=30)