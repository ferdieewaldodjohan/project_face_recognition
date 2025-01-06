import dlib
import cv2
import pickle
import numpy as np
import os

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directories containing known and augmented photos
known_directory = "known_faces"
augmented_directory = "augmented_faces"

# Initialize lists to store known encodings and names
known_encodings = []
known_names = []

# Function to extract face encoding from an image and get the name from the filename
def extract_face_encoding(image, filename, directory):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (1920, 1080))
    faces = face_detector(gray)
    if len(faces) == 0:
        return None, None
    face = faces[0]
    landmarks = shape_predictor(gray, face)
    encoding = np.array(face_recognizer.compute_face_descriptor(image, landmarks))
    
    # Extract name from filename (assuming filename is in the format "Name_Surname.jpg")
    name = os.path.basename(directory)
    
    return encoding, name

# Process known images and extract encodings
for root, dirs, files in os.walk(known_directory):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            encoding, name = extract_face_encoding(image, filename, root)
            if encoding is not None:
                known_encodings.append(encoding)
                known_names.append(name)

# Process augmented images and extract encodings
for root, dirs, files in os.walk(augmented_directory):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            encoding, name = extract_face_encoding(image, filename, root)
            if encoding is not None:
                known_encodings.append(encoding)
                known_names.append(name)


# Save updated encodings and names to a pickle file
with open("face_encodings_dlib.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

# Load updated encodings
with open("face_encodings_dlib.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Start video capture
video_capture = cv2.VideoCapture(0)

# Threshold for accuracy
accuracy_threshold = 30.0

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector(gray)


        if len(faces) == 1:
            face = faces[0]
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())

            # Validate face coordinates
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                print("Invalid face coordinates detected.")
                continue

            # Convert the detected face to grayscale
            face_gray = gray[y:y+h, x:x+w]
            
            # Get landmarks and face encoding
            landmarks = shape_predictor(face_gray, face)
            encoding = np.array(face_recognizer.compute_face_descriptor(frame, landmarks))

            # Compare encoding with known encodings
            distances = [np.linalg.norm(encoding - known_encoding) for known_encoding in known_encodings]
            min_distance = min(distances)
            match_index = distances.index(min_distance)
            name = known_names[match_index] if min_distance < 0.6 else "Unknown"

            # Calculate accuracy as a percentage
            accuracy = (1 - min_distance / 0.6) * 100 if name != "Unknown" else 0

            # Check if accuracy is below the threshold
            if accuracy < accuracy_threshold:
                name = "Unknown"
                color = (0, 0, 255)  # Red color
            else:
                color = (0, 255, 0)  # Green color

            # Draw a rectangle around the face
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())

            # Validate face coordinates
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                print("Invalid face coordinates detected.")
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Convert the detected face to grayscale
            face_gray = gray[y:y+h, x:x+w]

            # Validate grayscale face dimensions
            if face_gray.size == 0:
                print("Grayscale face image is empty.")
                continue

            # Display name and accuracy
            cv2.putText(frame, f"{name} ({accuracy:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Optionally, display the grayscale face in a separate window
            cv2.imshow(f"Grayscale Face - {name}", face_gray)

        # Display the video feed
        cv2.imshow("Face Recognition", frame)

        # Exit on 'q' key or when the window is closed
        if cv2.waitKey(1) & 0xFF == ord ("x") :
            break

        # Check if the window is closed
        if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    video_capture.release()
    cv2.destroyAllWindows()