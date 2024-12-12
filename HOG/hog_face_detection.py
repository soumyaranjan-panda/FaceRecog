import cv2

# Initialize HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

    # Resize the frame for better detection (optimize by reducing resolution for faster processing)
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to a standard resolution

    # Convert the frame to grayscale for better face detection
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detect faces using HOG
    faces, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.1)

    # Draw rectangle around detected faces and display confidence
    for (x, y, w, h) in faces:
        # Calculate confidence as the ratio of the area of the detected face to the total frame area
        confidence = (w * h) / (frame_resized.shape[0] * frame_resized.shape[1]) * 100  # In percentage

        # Draw rectangle around the face with color (0, 255, 0) and thickness of 2
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw a background for the confidence text to make it visible
        cv2.rectangle(frame_resized, (x, y - 25), (x + w, y), (0, 255, 0), -1)

        # Display confidence text with the background
        cv2.putText(frame_resized, f'Confidence: {confidence:.2f}%', 
                    (x + 5, y - 5),  # Slightly offset to avoid overlapping with the rectangle
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (255, 255, 255),  # White text color
                    2)

    # Display the frame with rectangles and confidence
    cv2.imshow("Live Face Detection with HOG", frame_resized)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
