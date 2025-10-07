#face.py
import os
import pickle
import logging

import cv2
import face_recognition
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def train_model(data_dir='faces', model_path='model.pkl'):
    known_encodings = []
    known_names = []

    # iterate folders like faces/Alice, faces/Bob
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model='hog')
            encs = face_recognition.face_encodings(rgb, boxes)

            for enc in encs:
                known_encodings.append(enc)
                known_names.append(person)

    if len(set(known_names)) < 2:
        raise ValueError("Need at least two different people to train.")

    # label-encode the string names
    le = LabelEncoder()
    labels = le.fit_transform(known_names)

    logging.info(f"Training on {len(known_encodings)} face samples.")
    clf = SVC(C=1.0, kernel='linear', probability=True)
    clf.fit(known_encodings, labels)

    # save the classifier + label encoder
    with open(model_path, 'wb') as f:
        pickle.dump({'classifier': clf, 'le': le}, f)

    logging.info(f"Model saved to '{model_path}'.")

def load_model(model_path='model.pkl'):
    """
    Load the face.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['classifier'], data['le']

def recognize_faces(frame, clf, le, trusted_set, threshold=0.7):
    """
    Detect faces, compute embeddings, classify them, and choose a color.
    If max prediction probability < threshold, labels as "Unknown".
    Returns a list of (l, t, r, b, name, color).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)

    boxes = face_recognition.face_locations(small, model='hog')
    annotations = []

    for box in boxes:
        top, right, bottom, left = [v * 2 for v in box]
        enc = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])[0]
        probs = clf.predict_proba([enc])[0]
        idx = probs.argmax()
        max_prob = probs[idx]

        if max_prob < threshold:
            name = "Unknown"
            color = (0, 0, 255)
        else:
            name = le.inverse_transform([idx])[0]
            color = (0, 255, 0) if name in trusted_set else (0, 0, 255)
        annotations.append((left, top, right, bottom, name, color))

    return annotations