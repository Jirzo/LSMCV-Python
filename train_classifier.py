import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el archivo pickle que contiene los datos y las etiquetas.
# Este archivo fue generado previamente con los datos procesados.
data_dict = pickle.load(open("./pickles/ABECEDARIO.pickle", "rb"))
print(type(data_dict))
print(data_dict)

# Inspección inicial de los datos para entender su estructura.
# Esto ayuda a verificar que los datos se cargaron correctamente.
# print(f"Tipo de data: {type(data_dict['data'])}")  # Confirmar que es una lista.
# print(f"Longitudes de las secuencias: {[len(item) for item in data_dict['data'][:5]]}")

# Normalización y padding:
# Dado que algunas secuencias pueden tener longitudes variables, necesitamos unificar su tamaño.
# Aquí utilizamos padding para que todas las secuencias tengan la misma longitud.
# max_len = max(
#     len(item) for item in data_dict["data"]
# )  # Longitud máxima entre todas las secuencias.
data = data_dict["data"]
labels = data_dict["labels"]

# Verificación para asegurar que cada muestra tenga una etiqueta correspondiente.
# Esto evita errores posteriores durante el entrenamiento.
assert len(data) == len(labels), "El número de datos y etiquetas no coincide."

# Dividir los datos en conjuntos de entrenamiento y prueba.
# - El conjunto de entrenamiento (80%) se utiliza para ajustar el modelo.
# - El conjunto de prueba (20%) se reserva para evaluar el rendimiento del modelo.
# - `shuffle=True` mezcla los datos antes de dividirlos.
# - `stratify=labels` asegura que la distribución de clases sea similar en ambos conjuntos.
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Entrenar un modelo Random Forest:
# RandomForestClassifier es un modelo de machine learning basado en árboles de decisión.
# Es robusto y puede manejar datos complejos con facilidad.
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluar el rendimiento del modelo utilizando precisión (accuracy).
# La precisión mide el porcentaje de predicciones correctas.
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"Precisión del modelo: {accuracy * 100}%")

f = open("model_abecedario.p", "wb")
pickle.dump({"model": model}, f)
f.close()
