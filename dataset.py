import cv2
import os

# Create a directory for storing the dataset
dataset_dir = "Dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize camera
cap = cv2.VideoCapture(0)

# Load face cascade
face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

# Counter for capturing images
count = 0

# Input user's name
name = input("Enter your name: ")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured face
        cv2.imwrite(f"{dataset_dir}/{name}_{count}.jpg", gray[y:y+h, x:x+w])

        # Display the image
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 30:  # Capture 30 images
        break

cap.release()
cv2.destroyAllWindows()
