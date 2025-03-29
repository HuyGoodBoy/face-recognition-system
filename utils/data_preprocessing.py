import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (100, 100))
            images.append(img.flatten())
            labels.append(filename.split("_")[0])  # Dùng tên file để tạo nhãn
    return np.array(images), labels

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100)).flatten()
    return img
