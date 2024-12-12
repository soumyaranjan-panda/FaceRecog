import cv2

def rescaleFrame(frame, scale = 0.2):
    # works for images, videos and live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Load Haar Cascade classifier
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the HOG face detector
hog_face_detector = cv2.HOGDescriptor()
hog_face_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the image
image_path = '1.jpg'  # Change this to the path of your image
image1 = cv2.imread(image_path)
image = rescaleFrame(image1)

if image is None:
    print("Error: Could not load image.")
    exit()

# Convert the image to grayscale for Haar Cascade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Haar Cascade Detection
haar_faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in haar_faces:
    confidence_haar = (w * h) / (image.shape[0] * image.shape[1]) * 100
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, f'Haar: {confidence_haar:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# HOG Detection (Using OpenCV's HOGDescriptor)
# Detect faces using HOG
boxes, weights = hog_face_detector.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)

for (x, y, w, h) in boxes:
    confidence_hog = (w * h) / (image.shape[0] * image.shape[1]) * 100
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f'HOG: {confidence_hog:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Display the image with both detection methods
cv2.imshow("HOG vs Haar Cascade Face Detection", image)

# Wait until a key is pressed and then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
