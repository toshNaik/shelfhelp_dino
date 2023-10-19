# script to go through each image in the dataset and convert it to feature representation using dino
from model_utils import Dino
import torch
import numpy as np
import os, sys


def batch_process(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

images = []
for root, subdir, imgs in os.walk('../dataset_v2'):
    for f in imgs:
        images.append(os.path.join(root, f))

dino = Dino()

for i, batch in enumerate(batch_process(images, batch_size=256)):
    feats = dino.extract_features(batch)
    print(f'Processed batch {i}')
    np.save(f'features/feat_writer_{i}', feats)