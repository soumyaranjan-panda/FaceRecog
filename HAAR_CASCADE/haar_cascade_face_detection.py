import cv2




# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam, or specify another index for external cameras

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle around detected faces and display confidence
    for (x, y, w, h) in faces:
        # Calculate confidence as the ratio of the area of the detected face to the total frame area
        confidence = (w * h) / (frame.shape[0] * frame.shape[1]) * 100  # In percentage

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display confidence text near the rectangle
        cv2.putText(frame, f'Confidence: {confidence:.2f}%', 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (0, 255, 0), 
                    2)

    # Display the frame with rectangles and confidence
    cv2.imshow("Live Face Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
