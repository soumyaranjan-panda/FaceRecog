import cv2

# Load Haar Cascade classifier
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the HOG face detector
hog_face_detector = cv2.HOGDescriptor()
hog_face_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start webcam capture
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar Cascade Detection
    haar_faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in haar_faces:
        confidence_haar = (w * h) / (frame.shape[0] * frame.shape[1]) * 100
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Haar: {confidence_haar:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # HOG Detection (Using OpenCV's HOGDescriptor)
    # Detect faces using HOG
    boxes, weights = hog_face_detector.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in boxes:
        confidence_hog = (w * h) / (frame.shape[0] * frame.shape[1]) * 100
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'HOG: {confidence_hog:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame with both detection methods
    cv2.imshow("HOG vs Haar Cascade Face Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
