from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

def train_eigenfaces(X_train, n_components=50):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    pca = PCA(n_components=n_components).fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)

    with open('models/eigenfaces_model.pkl', 'wb') as f:
        pickle.dump(pca, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_train_pca, scaler, pca
