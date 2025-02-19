import os
import cv2
import numpy as np
from tqdm import tqdm 
from joblib import dump 
from settings.collect_image import DATA_DIR

def datasetcCreation(hands_model):
    data = []
    labels = []
    class_names = sorted(os.listdir(DATA_DIR))
    class_dict = {class_names: idx for idx, class_names in enumerate(class_names)}
    skipped_images_count = {class_name: 0 for class_name in class_names}
    for dir_ in tqdm(os.listdir(DATA_DIR), desc="Proccesing classes"):
        class_path = os.path.join(DATA_DIR, dir_)

        # Check if directories exist
        if not os.path.isdir(class_path):
            continue

        # Process each image in the current directory
        for img_path in tqdm(os.listdir(class_path), desc=f"Proccesing images from {dir_}", leave=False):
            img_full_path = os.path.join(class_path, img_path)

            # Read and validate the image
            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Warning: Image could not be loaded {img_full_path}. Skipping...")
                skipped_images_count[dir_] += 1
                continue
            
            # Convert image to RGB to MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe to detect landmarks
            result = hands_model.process(img_rgb)
            if result.multi_hand_landmarks:  # If hands are detected
                for hand_landmarks in result.multi_hand_landmarks:
                    data_aux = []
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                    # Add landmarks and tags to main lists
                    data.append(data_aux)
                    labels.append(class_dict[dir_])  # Use numeric index of the class
            else:
                print(f"Warning: No landmarks detected in {img_full_path}. Skipping...")

                skipped_images_count[dir_] += 1

    # Convert to NumPy arrays
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Save the dataset as a file using joblib
    output_file = "datalettersset.joblib"
    dump({"data": data, "labels": labels}, output_file)

    # Show summary of skipped images
    print("\nResumen de imágenes saltadas por clase:")
    for class_name, count in skipped_images_count.items():
        print(f"{class_name}: {count} imágenes saltadas")


