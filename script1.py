import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces or features in the image
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []  # Storing the coordinates of the detected feature
    for (x, y, w, h) in features:
        # Draw rectangle around the detected feature (e.g., face or eyes)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        # Put the text above the rectangle
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, img

def detect(img, faceCascade, eyeCascade):
    # Define colors for the rectangle and text
    colors = {"blue": (255, 0, 0), "red": (0, 0, 255)}

    # Detect face in the image
    face_coords, img = draw_boundary(img, faceCascade, 1.1, 10, colors['blue'], "Face")

    # If a face is detected, detect the eyes within the face region
    if len(face_coords) == 4:
        x, y, w, h = face_coords  # Coordinates of the face
        # Define the region of interest (ROI) to detect eyes within the face
        roi_img = img[y:y+h, x:x+w]
        
        # Detect eyes in the face region (ROI)
        draw_boundary(roi_img, eyeCascade, 1.1, 14, colors['red'], "Eyes")
      
    return img

# Load the Haar cascade for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Open the video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()  # Reading video frame
    if not ret:  # Check if the frame is read correctly
        break

    # Detect faces and eyes in the frame
    img = detect(img, face_cascade, eye_cascade)

    # Display the frame with face and eye detection
    cv2.imshow("Face and Eye Detection", img)
    
    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
