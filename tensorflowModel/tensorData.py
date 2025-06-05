import tensorflow as tf

# Parámetros esperados
timesteps = 10  # Número de frames por secuencia
num_landmarks = 21
features_per_landmark = 3

# Función para parsear el TFRecord
def _parse_function(example_proto):
    feature_description = {
        "landmarks": tf.io.FixedLenFeature([timesteps * num_landmarks * features_per_landmark], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
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
