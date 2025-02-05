from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import numpy as np

def randomForestClassifier():
    # Cargar datos
    data_dict = load("datalettersset.joblib")
    data, labels = data_dict["data"], data_dict["labels"]

    # Normalización
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Dividir datos
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    # === Random Forest con GridSearchCV ===
    rf_params = {
        'n_estimators': [1000],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    print("Random Forest Optimization...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_params, cv=3, verbose=2, n_jobs=-1)
    rf_grid.fit(x_train, y_train)

    best_rf_model = rf_grid.best_estimator_
    print(f"Best Random Forest Hyperparameters: {rf_grid.best_params_}")

    # Evaluación del modelo Random - Forest - Donde almacenar el codigo - Branches.
    rf_y_predict = best_rf_model.predict(x_test)
    rf_accuracy = accuracy_score(y_test, rf_y_predict)
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_y_predict))

    # Guardar modelo y escalador para uso futuro
    dump({"model": best_rf_model, "scaler": scaler}, "randomFT_model.joblib")
    print("Best Random Forest model saved as 'randomFT_model.joblib'")