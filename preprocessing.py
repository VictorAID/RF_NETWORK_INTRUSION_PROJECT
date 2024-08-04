import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def normalize_features(data, features):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    return data_scaled, scaler

def reduce_dimensionality(data_scaled, variance_retained=0.95):
    pca = PCA(n_components=variance_retained)
    X_pca = pca.fit_transform(data_scaled)
    return X_pca, pca

def label_anomalies(X_pca):
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    return np.where(labels == -1, 1, 0)
