from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Para mostrar el progreso en terminal
from joblib import dump, load

def xgboostClassifier():
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
    # === XGBoost ===
    print("Entrenando modelo XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    with tqdm(total=xgb_model.n_estimators, desc="Entrenando XGBoost") as pbar:
        def update_progress(current_round, total_rounds):
            pbar.update(current_round - pbar.n)

        xgb_model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            callbacks=[update_progress]
        )

    # Evaluación del modelo XGBoost
    xgb_y_predict = xgb_model.predict(x_test)
    xgb_accuracy = accuracy_score(y_test, xgb_y_predict)
    print(f"Precisión XGBoost: {xgb_accuracy * 100:.2f}%")
    print("\nReporte de clasificación XGBoost:")
    print(classification_report(y_test, xgb_y_predict))

    # Guardar modelo XGBoost
    dump({"model": xgb_model, "scaler": scaler}, "xgb_model.joblib")
    print("Modelo XGBoost guardado como 'xgb_model.joblib'")