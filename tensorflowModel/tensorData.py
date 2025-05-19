import tensorflow as tf
import joblib
import numpy as np

# Ruta al archivo joblib que contiene mi dataset
joblib_file_route = 'datalettersset.joblib'

# Cargar los datos del archivos joblib
loaded_dataset = joblib.load(joblib_file_route)
landmark_coordinates = loaded_dataset['data']
labels = loaded_dataset['labels']

# Crear el tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((landmark_coordinates, labels))

# Configurar el dataset para el entrenamiento (batching, shuffling, prefetching)
batch_size = 32
dataset = dataset.shuffle(buffer_size=len(landmark_coordinates)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Ahora puedes usar este 'dataset' para entrenar tu modelo de TensorFlow
# Por ejemplo:
# model.fit(dataset, epochs=10)

# Iterar sobre el dataset para verificar los datos
for cordinates, labels in dataset:
    print("Lote de coordenadas:", cordinates.numpy().shape)
    print("Lote de etiquetas:", labels.numpy().shape)
