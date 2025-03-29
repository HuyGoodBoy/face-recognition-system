import cv2
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from utils.data_preprocessing import preprocess_image, load_images_from_folder
import numpy as np

face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt",
                                    "models/res10_300x300_ssd_iter_140000.caffemodel")

with open('models/eigenfaces_model.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

folder = 'data/training_images'
X_train, y_train = load_images_from_folder(folder)
X_train_scaled = scaler.transform(X_train)
X_train_pca = pca.transform(X_train_scaled)


def detect_face(image_path):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    largest_face = None
    largest_area = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            area = (endX - startX) * (endY - startY)
            if area > largest_area:
                largest_face = image[startY:endY, startX:endX]
                largest_area = area
    return largest_face


def recognize_face(image_path):
    face = detect_face(image_path)
    if face is None:
        return "Không tìm thấy khuôn mặt"

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (100, 100)).flatten()
    face_scaled = scaler.transform([face])
    face_pca = pca.transform(face_scaled)

    distances = euclidean_distances(face_pca, X_train_pca)
    min_distance = np.min(distances)
    min_index = np.argmin(distances)

    threshold = 5000
    if min_distance < threshold:
        return y_train[min_index]
    else:
        return "Không xác định"
