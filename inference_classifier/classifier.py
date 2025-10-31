import cv2
import numpy as np
from joblib import load
import tensorflow as tf
from collections import deque
from settings.landmarks import mp_drawing_styles, mp_hands, mp_drawing


def iClassifier(hands):

    window_title = "Inference LSTM"
    cap = cv2.VideoCapture(1)

    # Cargar modelo y escalador
    model_dict = load("randomFT_model.joblib")
    print(model_dict)
    model = model_dict["model"]
    scaler = model_dict["scaler"]

    try:
        # Cargar modelo LSTM de TensorFlow (Keras)
        model = tf.keras.models.load_model("mejor_modelo.keras")
        # Cargar el escalador.
        # Este escalador debe de mcontar con 63 características (x, y, z)
        scaler = load("mi_scaler_xyz.joblib")

    except Exception as e:
        print(f"Error al cargar modelo o scaler: {e}")
        print("Asegúrate de tener 'mejor_modelo.keras' y 'mi_scaler_xyz.joblib'")
        return
    
    # --- ### Parámetros de Secuencia ---
    SEQUENCE_LENGTH = 30 # Debe coincidir con el entrenamiento
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # Esta variable la usaremos para mostrar la prediccion de forma fluida
    predicted_character = ""

    # Lista de etiquetas (asegúrate de que tenga 28 elementos, 
    # igual que la capa final de tu modelo)
    alphabet = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'LL', 'M', 'N', 'Ñ',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    ]

    # Diccionario de etiquetas
    labels_dict = {i: letter for i, letter in enumerate(alphabet)}

    # --- ### Características Esperadas ---
    expected_features = 63 # 21 landmarks * 3 (x,y,z)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Please check the camera.")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # --- ### Extracción de x, y, z ---
                data_aux = []
                x_coords, y_coords = [], [] # Para el bounding box
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                    data_aux.extend([landmark.x, landmark.y, landmark.z])
                
                # Calcular el bounding box
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10
                x2 = int(max(x_coords) * W) + 10
                y2 = int(max(y_coords) * H) + 10

                # --- ### Lógica de Secuencia y Predicción ---
                # if len(data_aux) < expected_features:
                #     data_aux.extend([0] * (expected_features - len(data_aux)))

                if len(data_aux) == expected_features:
                    # 1. Escalar las características (debe ser 2D para el scaler)
                    data_scaled = scaler.transform(np.arrat(data_aux).reshape(1, -1))

                    # 2. Añadir el frame escalado (vector de 63,) al buffer
                    sequence_buffer.append(data_scaled[0])

                    # 3. Predecir SÓLO si el buffer está lleno
                    if len(sequence_buffer) == SEQUENCE_LENGTH:
                        # Prepara la secuencia (1, 20, 63)
                        input_data = np.expand_dims(np.array(list(sequence_buffer)), axis=0)

                        # Realiza la prediccion
                        prediction_probs = model.predict(input_data)

                        # Obtien la clase con mayor probabilidad
                        predicted_index = np.argmax(prediction_probs[0])

                        # Actualiza la variable de texto
                        predicted_character = labels_dict[predicted_index]

                    # Mostrar resultado en la ventana
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        predicted_character,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                else:
                    print("Error: Número de características no coincide con el modelo.")
        else:
            #  Limpiar el buffer si no se detecta mano
            sequence_buffer.clear()
            predicted_character = ""
            cv2.putText(
                frame, "No se detectaron manos.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA,
            )

        # Mostrar imagen procesada
        cv2.imshow(window_title, frame)

        # Salir con 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
