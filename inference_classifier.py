import os
import cv2
import pickle

import numpy as np
import mediapipe as mp

from joblib import dump, load

cap = cv2.VideoCapture(1)
model_dict = pickle.load(open("model_abecedario.p", "rb"))
model = model_dict["model"]

# Initialize MediaPipe for hand detection.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {i: chr(65 + i) for i in range(26)}  # Esto crea un diccionario {0: 'A', 1: 'B', ..., 25: 'Z'}

expected_features = 42  # Adjust based on the model's input feature count.

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
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

        # Collect features for all detected hands.
        for hand_landmarks in result.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Pad `data_aux` if the number of detected landmarks is less than expected.
        if len(data_aux) < expected_features:
            data_aux.extend([0] * (expected_features - len(data_aux)))

        # Ensure `data_aux` matches the expected feature count.
        if len(data_aux) == expected_features:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            print(predicted_character)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(
                frame,
                predicted_character,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
        else:
            print("Feature mismatch: Check input data collection.")
    else:
        print("No hands detected.")

    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()