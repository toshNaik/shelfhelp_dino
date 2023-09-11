import os
from extract_feats import extract_features, load_pretrained_weights
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from main import dino_load
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


vits8 = dino_load()
# cereal_imgs = [os.path.join('data', 'cereals', i) for i in os.listdir('data/cereals')]
# soda_imgs = [os.path.join('data', 'soda', i) for i in os.listdir('data/soda')]
# cereal_feats = extract_features(vits8, cereal_imgs)
# soda_feats = extract_features(vits8, soda_imgs)
features = []
for i in os.listdir('features/'):
    if not i.startswith('feature_shelf'):
        continue
    feature = os.path.join('features', i)
    feature = np.load(feature)
    print(feature.shape)
    features.append(feature)
X = np.concatenate(features, axis=0)
print(X.shape)

# Calculate inertia for different k
inertias = []
for k in range(1, 40):
    kmeans = KMeans(n_clusters=k).fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1, 40), inertias)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# # Cluster into 5 clusters
# kmeans = KMeans(n_clusters=5).fit(X)
# y_pred = kmeans.predict(X)

# # Reduce to 2D for plotting
# pca = PCA(n_components=2).fit(X)
# X_pca = pca.transform(X)

# # Plot clusters in 2D
# plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred) 
# plt.title('KMeans Clustering')
# plt.show()

# centroids = kmeans.cluster_centers_
# print(centroids)

# labels = kmeans.labels_
# print(labels)

# Cluster feature vectors
kmeans = KMeans(n_clusters=10).fit(X) 

# Generate 3D t-SNE projection
tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(X)

# Plot clusters in 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_tsne[:,0], X_tsne[:,1], X_tsne[:,2], c=kmeans.labels_)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Generate 2D t-SNE projection
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Plot clusters in 2D scatter plot
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=kmeans.labels_)
plt.show()