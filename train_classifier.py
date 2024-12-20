from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle
import numpy as np

# Cargar datos
data_dict = pickle.load(open("data_tensorflow.pickle", "rb"))
data = data_dict["data"]
labels = data_dict["labels"]

# Normalización
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Dividir datos
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(x_train, y_train)

# Evaluar
y_predict = model.predict(x_test)
print(f"Precisión: {accuracy_score(y_test, y_predict) * 100}%")
print("Reporte de clasificación:")
print(classification_report(y_test, y_predict))

# Guardar modelo
dump(model, "model_abecedario.joblib")
