import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle

def load_data():
    '''
    Utility function for evaluate to load all npy files into a single dataset
    '''
    data_dir = 'features/'
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    all_data = []
    for file in files:
        data = np.load(os.path.join(data_dir, file))
        all_data.append(data)
        
    all_data = np.vstack(all_data)

    # 2. Reshape data if needed
    # Assuming you want to treat each 384-length vector independently
    all_data = all_data.reshape(-1, 384)
    return all_data

def cluster(num_clusters=30):
    '''
    Cluster for num_clusters and save the model
    '''
    all_data = load_data()
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(all_data)

    cluster_centers = kmeans.cluster_centers_

    with open('kmeans_models/kmeans_fitted30.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
def evaluate():
    '''
    Evaluate for various cluster sizes using the elbow method and silhouette scores
    '''
    all_data = load_data()

    # 3. Apply KMeans clustering and use the elbow method
    distortions = []
    K = range(10, 300)  # Change the range if needed
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(all_data)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('Elbow.png')

    # 4. Compute silhouette scores
    sil_scores = []
    K_range = range(10, 300)  # Starts from 2 as silhouette score requires at least 2 clusters
    for k in K_range:
        kmeanModel = KMeans(n_clusters=k)
        cluster_labels = kmeanModel.fit_predict(all_data)
        silhouette_avg = silhouette_score(all_data, cluster_labels)
        sil_scores.append(silhouette_avg)

    plt.figure(figsize=(10, 6))
    plt.plot(K_range, sil_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis For Optimal k')
    plt.savefig('Silhouette.png')


if __name__ == '__main__':
    cluster(num_clusters=30)