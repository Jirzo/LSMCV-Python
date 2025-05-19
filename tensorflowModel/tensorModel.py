import tensorflow as tf


N_FRAMES = 30
FEATURES_PER_FRAME = 42
NUM_CLASSES = 21  # A-Z + Ñ (ajústalo a tu caso)

def parse_tfrecord(proto):
    feature_description = {
        'landmarks': tf.io.FixedLenFeature([N_FRAMES * FEATURES_PER_FRAME], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_example = tf.io.parse_single_example(proto, feature_description)

    landmarks = tf.reshape(parsed_example['landmarks'], (N_FRAMES, FEATURES_PER_FRAME))
    label = tf.one_hot(parsed_example['label'], depth=NUM_CLASSES)  # opcional
    return landmarks, label

# raw_dataset = tf.data.TFRecordDataset('landmarks_dataset')
# parsed_dataset = raw_dataset.map(parse_tfrecord)
# parsed_dataset = parsed_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

def load_dataset(tfrecord_path, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = load_dataset("sequences_dataset.tfrecord")

model = tf.keras.Sequential([
    tf.layers.Input(shape=(N_FRAMES, FEATURES_PER_FRAME)),

    # LSTM puede ser bidireccional para capturar patrones hacia adelante y atrás
    tf.layers.Bidirectional(tf.layers.LSTM(64, return_sequences=True)),
    tf.layers.Bidirectional(tf.layers.LSTM(32)),

    tf.layers.Dense(64, activation='relu'),
    tf.layers.Dropout(0.4),
    tf.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)