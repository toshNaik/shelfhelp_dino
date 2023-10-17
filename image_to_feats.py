# script to go through each image in the dataset and convert it to feature representation using dino
from extract_feats import load_pretrained_weights, extract_features
import torch
import numpy as np
import os, sys

def dino_load():
    vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    load_pretrained_weights(vits8, 'dino_checkpoint.pth', 'teacher')
    # need to be popped since when calling torch.hub.load. if other model dir has a same named .py file it causes conflicts. Here caused by utils.py present in both dino and yolo directories
    # https://github.com/pytorch/hub/issues/243
    sys.modules.pop('utils')
    vits8.to(torch.device('cuda'))
    return vits8

def batch_process(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

images = []
for root, subdir, imgs in os.walk('../dataset_v2'):
    for f in imgs:
        images.append(os.path.join(root, f))

dino = dino_load()

for i, batch in enumerate(batch_process(images, batch_size=256)):
    feats = extract_features(dino, batch)
    print(f'Processed batch {i}')
    np.save(f'features/feat_writer_{i}', feats)