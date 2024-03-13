import os
import cv2
import numpy as np

# Load face dataset
dataset_dir = "Dataset"
faces = []
labels = []

label_dict = {}  # Dictionary to map labels to integers
current_label = 0

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            if label not in label_dict:
                label_dict[label] = current_label
                current_label += 1
            label_id = label_dict[label]
            img = cv2.imread(path, 0)
            faces.append(img)
            labels.append(label_id)

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save trained model
recognizer.save("trained_model.yml")
