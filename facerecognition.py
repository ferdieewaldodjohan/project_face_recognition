import cv2
import dlib
import pickle
import os
import numpy as np
from tqdm import tqdm

# Directories for known and augmented faces
KNOWN_FACES_DIR = ["known_faces", "../augmented_faces"]

# Load dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Lists for storing encodings and names
encodings = []
names = []

# Loop through all the images in the directories
for dir in KNOWN_FACES_DIR:
    for folder in tqdm(os.listdir(dir), desc=f"Processing folders in {dir}"):
        folder_path = os.path.join(dir, folder)
        for filename in tqdm(os.listdir(folder_path), desc=f"Processing images in {folder}", leave=False):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (640, 640))  # Resize image to a consistent size

            # Detect faces in the image
            faces = face_detector(image)
            for face in faces: 
                # Detect landmarks for the cropped face
                landmarks = shape_predictor(image, face)

                # Compute face descriptor (encoding) for the cropped face
                encoding = np.array(face_recognizer.compute_face_descriptor(image, landmarks))

                # Append encoding and name
                encodings.append(encoding)
                names.append(folder)

# Save the face encodings to a file
with open("face_encodings_dlib.pkl", "wb") as f:
    pickle.dump((encodings, names), f)

print("Face encodings saved successfully.")
