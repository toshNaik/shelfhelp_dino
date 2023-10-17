import os, cv2
import sys
import numpy as np
from extract_feats import load_pretrained_weights, extract_features
import torch
from sklearn.cluster import KMeans
import pandas as pd
from PIL import Image
import pickle
import matplotlib.pyplot as plt
# do not import utils here if one is made look dino_load for issue

def dino_load():
    vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    load_pretrained_weights(vits8, 'dino_checkpoint.pth', 'teacher')
    # need to be popped since when calling torch.hub.load. if other model dir has a same named .py file it causes conflicts. Here caused by utils.py present in both dino and yolo directories
    # https://github.com/pytorch/hub/issues/243
    sys.modules.pop('utils')
    return vits8

def yolo_load():
    yolo = torch.hub.load("ultralytics/yolov5", "custom", "best_100.pt")
    sys.modules.pop('utils')
    return yolo

def main():
    # load dino
    vits8 = dino_load().to('cuda')
    # load yolo
    yolo_model = yolo_load()

    # load clusters
    with open('kmeans_fitted50.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    
    # load images
    for img in os.listdir('shelf_images'):
        frame = cv2.imread(os.path.join('shelf_images', img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # run yolo
        pred = yolo_model(frame.copy())
        # extract patches
        table = pred.pandas().xyxy[0]
        xmin = table['xmin'].tolist()
        xmax = table['xmax'].tolist()
        ymin = table['ymin'].tolist()
        ymax = table['ymax'].tolist()
        scores = table['confidence'].tolist()
        crops = []
        # get patches
        for idx, patch in enumerate(zip(xmin, xmax, ymin, ymax, scores)):
            if patch[4] > 0.65:
                x1, x2, y1, y2 = int(patch[0]), int(patch[1]), int(patch[2]), int(patch[3])
                crop = frame[y1:y2, x1:x2]
                pil_img = Image.fromarray(crop)
                ## uncomment this to save patches and view them 
                # pil_img.save('cropped/'+img.split('.')[0]+"_"+str(idx)+'.jpeg')
                crops.append(pil_img)
        # get features
        features = extract_features(vits8, crops)
        print(features.shape)
        if features.size == 0:
            continue
        # cluster all features and get distribution as x*c1 + y*c2 + z*c3 + ... where c1 is cluster 1 center
        clusters = kmeans.predict(features)
        # get distribution
        dist = np.zeros(50)
        for i in clusters:
            dist[i] += 1
        # plot as histogram
        plt.figure()
        plt.bar(range(50), dist)
        plt.savefig('shelf_images_hist/'+img.split('.')[0]+'.png')
        print(dist)

# def main():
#     # load dino
#     vits8 = dino_load()
#     # load yolo
#     yolo_model = yolo_load()

#     for img in os.listdir('shelf_images'):
#         if not img.startswith('shelf'):
#             continue
#         print(img)
#         frame = cv2.imread(os.path.join('shelf_images', img))
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # run yolo
#         pred = yolo_model(frame.copy())
#         pred.show()
#         # strange way to get pd table but found here: https://github.com/ultralytics/yolov5/issues/7651
#         table = pred.pandas().xyxy[0]
#         xmin = table['xmin'].tolist()
#         xmax = table['xmax'].tolist()
#         ymin = table['ymin'].tolist()
#         ymax = table['ymax'].tolist()
#         scores = table['confidence'].tolist()
#         crops = []
#         # get patches
#         for idx, patch in enumerate(zip(xmin, xmax, ymin, ymax, scores)):
#             if patch[4] > 0.65:
#                 x1, x2, y1, y2 = int(patch[0]), int(patch[1]), int(patch[2]), int(patch[3])
#                 crop = frame[y1:y2, x1:x2]
#                 pil_img = Image.fromarray(crop)
#                 pil_img.save('cropped/'+img.split('.')[0]+"_"+str(idx)+'.jpeg')
#                 crops.append(pil_img)
#         # get features
#         features = extract_features(vits8, crops)
#         # save features for now
#         if features.size != 0:
#             np.save('features/feature_'+img.split('.')[0], features)

if __name__ == '__main__':
    main()