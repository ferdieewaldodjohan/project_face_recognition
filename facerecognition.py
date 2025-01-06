import cv2
import dlib
import pickle
import os
import numpy as np

KNOWN_FACES_DIR = "known_faces"

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

encodings = []
names = []

def convert_81_to_68(landmarks_81):
    mapping = list(range(68))
    return dlib.full_object_detection(
        landmarks_81.rect,
        dlib.points([landmarks_81.part(i) for i in mapping])
    )

for folder in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, folder)):
        image_path = os.path.join(KNOWN_FACES_DIR, folder, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (1920, 1080))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Image type: {type(gray)}, Shape: {gray.shape}, Dtype: {gray.dtype}")

        cv2.imwrite(os.path.join(KNOWN_FACES_DIR, folder, f"{os.path.splitext(filename)[0]}_cut.jpg"), gray)
        faces = face_detector(gray)
        for face in faces:
            landmarks_81 = shape_predictor(gray, face)

            landmarks_68 = convert_81_to_68(landmarks_81)

            encoding = np.array(face_recognizer.compute_face_descriptor(image, landmarks_68))
            encodings.append(encoding)
            names.append(os.path.splitext(filename)[0])

with open("face_encodings_dlib.pkl", "wb") as f:
    pickle.dump((encodings, names), f)
print("Encodings saved.")
