import cv2
import os
import json

# Initialize camera
cap = cv2.VideoCapture(0)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

# Load face cascade
face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

# Load metadata (folder names)
with open("metadata.json", "r") as f:
    folder_names = json.load(f)

# Create directory for storing captured images
images_dir = "Images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        # Recognize the face
        label, confidence = recognizer.predict(roi_gray)
        if confidence < 70:  # You may need to adjust the threshold
            recognized_name = folder_names[label]
            cv2.putText(frame, f"Recognized: {recognized_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Capture an image
        filename = os.path.join(images_dir, f"captured_image_{len(os.listdir(images_dir))}.jpg")
        cv2.imwrite(filename, frame)

cap.release()
cv2.destroyAllWindows()
