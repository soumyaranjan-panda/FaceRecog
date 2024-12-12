import cv2
import os
import numpy as np

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to training images (Make sure you have a dataset of faces)
path = "training_faces/"  # Update the path to your dataset

# Function to get images and labels for training
def get_images_and_labels(path):
    faces = []
    labels = []
    label_to_name = {}  # Dictionary to map label numbers to folder names
    current_label = 0

    for dir_name in os.listdir(path):
        subject_path = os.path.join(path, dir_name)
        if os.path.isdir(subject_path):
            label_to_name[current_label] = dir_name  # Map the label to the folder name
            for filename in os.listdir(subject_path):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(subject_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # Detect faces using the Haar Cascade face detector
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    print(f"Detecting faces in {img_path}...")
                    if len(faces_detected) > 0:
                        print(f"Faces detected: {len(faces_detected)}")
                        for (x, y, w, h) in faces_detected:
                            face = img[y:y + h, x:x + w]
                            faces.append(face)
                            labels.append(current_label)
                    else:
                        print(f"No faces detected in {img_path}")  # Debug line
            current_label += 1
    return faces, labels, label_to_name

# Training the recognizer
faces, labels, label_to_name = get_images_and_labels(path)

if len(faces) == 0 or len(labels) == 0:
    print("Error: No faces found for training.")
else:
    recognizer.train(faces, np.array(labels))
    recognizer.save("face_recognizer.yml")
    print("Model trained and saved as 'face_recognizer.yml'")

# Now for live face recognition (from webcam)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Recognize faces
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face)

        # Get the folder name (person's name) using the label
        person_name = label_to_name.get(label, "Unknown")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
