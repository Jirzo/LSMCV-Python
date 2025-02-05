import os
import cv2
from dataset.create_dataset import datasetcCreation
from hands_detection.landmarks import mediapipe_detection_fn
from image_collection.collect_img import frame_instuctions
from training.randomForestTrainer import randomForestClassifier
from inference_classifier.classifier import iClassifier
from settings.landmarks import mp_hands, mp_face_mesh, mp_drawing

hands_model = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Options menu
print("Select the action you want to perform:")
print("1: Collecting images")
print("2: Create dataset")
print("3: Entrenar modelo")
print("4: Train model")

# Windows title
window_title = ""

try:
    option_selected = int(input("Select an option (1-4): "))
    if option_selected not in [1, 2, 3, 4]:
        raise ValueError("Invalid option")
except ValueError:
    print("Error: You must enter a number between 1 and 4.")
    exit()

# Ejecución de la opción seleccionada
if option_selected == 1:
    print("Starting image collection...")
    window_title = "Collect Images"
    frame_instuctions()
    exit()  # Salir después de ejecutar

elif option_selected == 2:
    print("Creating dataset...")
    datasetcCreation(hands_model)  # Pasa el modelo si es necesario
    exit()

elif option_selected == 3:
    print("Training model...")
    randomForestClassifier()
    exit()

elif option_selected == 4:
    print("Running classifier inference...")
    window_title = "Inference"
    # Configuración de la cámara y modelos
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Please check the camera.")
            break

        # Detección con Mediapipe
        image, result = mediapipe_detection_fn(frame, hands_model)
        iClassifier(hands_model, frame)  # Ejecutar inferencia

        # Mostrar imagen procesada
        cv2.imshow(window_title, image)

        # Salir con 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
