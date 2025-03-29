import os
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.data_preprocessing import load_images_from_folder

# Đường dẫn đến thư mục chứa ảnh huấn luyện
folder = 'data/training_images'

print("Đang tải ảnh từ thư mục huấn luyện...")
X_train, y_train = load_images_from_folder(folder)

if X_train.size == 0:
    print("Không tìm thấy ảnh huấn luyện trong thư mục 'training_images'. Vui lòng thêm ảnh và thử lại.")
    exit()

print("Đang chuẩn hóa dữ liệu ảnh...")
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

print("Đang huấn luyện mô hình PCA để tạo Eigenfaces...")
pca = PCA(n_components=9).fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)

with open('models/eigenfaces_model.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Đã lưu mô hình Eigenfaces và scaler thành công!")
