from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import pickle
import numpy as np


# Cargar datos
data_dict = load("data_tensorflow.joblib")
data, labels = data_dict["data"], data_dict["labels"]

# Normalización
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Dividir datos
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Crear y entrenar modelo Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(x_train, y_train)

# Validación cruzada (opcional, pero recomendable para evaluar estabilidad)
cv_scores = cross_val_score(model, data, labels, cv=5)
print(f"Validación cruzada (5 folds): {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# Evaluación en el conjunto de prueba
y_predict = model.predict(x_test)
print(f"Precisión en prueba: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_predict))

# Matriz de confusión
print("\nMatriz de confusion:")
print(confusion_matrix(y_test, y_predict), y_predict)

# Guardar modelo y escalador para uso futuro
dump({"model": model, "scaler": scaler}, "model_abecedario.joblib")
print("Modelo y escalador guardados como 'model_abecedario.joblib'")
