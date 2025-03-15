import cv2
import numpy as np
from joblib import load
from settings.landmarks import mp_drawing_styles, mp_hands, mp_drawing


def iClassifier(hands):

    window_title = "Inference"
    cap = cv2.VideoCapture(1)

    # Cargar modelo y escalador
    model_dict = load("randomFT_model.joblib")
    print(model_dict)
    model = model_dict["model"]
    scaler = model_dict["scaler"]

    alphabet = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N',
        'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    ]

    # Diccionario de etiquetas
    labels_dict = {i: letter for i, letter in enumerate(alphabet)}
    print(labels_dict)
    expected_features = 42

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
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extraer coordenadas de la mano detectada
                data_aux = []
                x_coords, y_coords, z_coords = [], [], []
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                    data_aux.extend([landmark.x, landmark.y])
                
                # Calcular el bounding box
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10
                x2 = int(max(x_coords) * W) + 10
                y2 = int(max(y_coords) * H) + 10

                # Validar el número de características
                if len(data_aux) < expected_features:
                    data_aux.extend([0] * (expected_features - len(data_aux)))

                if len(data_aux) == expected_features:
                    # Escalar las características y predecir
                    data_aux = np.array(data_aux).reshape(1, -1)
                    data_aux = scaler.transform(data_aux)
                    prediction = model.predict(data_aux)
                    predicted_character = labels_dict[int(prediction[0])]
                    # print(data_aux)
                    # print(prediction)
                    # print(predicted_character)

                    # Mostrar resultado en la ventana
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        predicted_character,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    print("Error: Número de características no coincide con el modelo.")
        else:
            cv2.putText(
                frame,
                "No se detectaron manos.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Mostrar imagen procesada
        cv2.imshow(window_title, frame)

        # Salir con 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
