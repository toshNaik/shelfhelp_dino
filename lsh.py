import numpy as np
import os

class LSHashing:
    def __init__(self, num_hashes: int, num_buckets: int, dimensions: int) -> None:
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.dimensions = dimensions
        self.hash_functions = self._generate_hash_fns()
    
    def _generate_hash_fns(self):
        hashes = []
        for i in range(self.num_hashes):
            random_vector = np.random.randn(self.dimensions)
            hashes.append(random_vector)
        return hashes

    def _hash(self, hash_fn, vec):
        cosine_dist = 1 - np.dot(vec, hash_fn) / (np.linalg.norm(vec)*np.linalg.norm(hash_fn)) 
        return cosine_dist
    
    def hash_vector(self, vec):
        hash_value = [round(self._hash(hf, vec), 4) for hf in self.hash_functions]
        return hash_value

if __name__ == '__main__':
    features = []
    for i in os.listdir('features/'):
        feature = os.path.join('features', i)
        feature = np.load(feature)
        print(feature.shape)
        features.append(feature)
    features = np.concatenate(features, axis=0)

    lsh = LSHashing(5, 5, 384)
    for vector in features:
        h = lsh.hash_vector(vector)
        print(h)