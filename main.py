import os, cv2
import sys
import numpy as np
from model_utils import Dino, yolo_load, yolo_forward
import torch
from sklearn.cluster import KMeans
import pandas as pd
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Affine2D
# do not import utils here if one is made look dino_load function for issue

def dist_plot():
    '''
    Plots distribution of clusters for each image in shelf_images
    '''
    # load dino
    vits8 = Dino()
    # load yolo
    yolo_model = yolo_load()

    # load kmeans model
    num_clusters = 30
    with open(f'kmeans_models/kmeans_fitted{num_clusters}.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    
    # load shelf_images
    for img in os.listdir('shelf_images'):
        frame = cv2.imread(os.path.join('shelf_images', img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # run yolo to get patches
        crops = yolo_forward(yolo_model, frame)
        # get features
        features = vits8.extract_features(crops)
        if features.size == 0:
            continue
        # cluster all features and get distribution as x*c1 + y*c2 + z*c3 + ... where c1 is cluster 1 center
        clusters = kmeans.predict(features)
        dist = np.bincount(clusters, minlength=num_clusters)

        # Group crops based on their cluster assignments
        cluster_images = [[] for _ in range(num_clusters)]
        for i, cluster_idx in enumerate(clusters):
            cluster_images[cluster_idx].append(crops[i])

        # plot histogram and images
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(25,10), gridspec_kw={'height_ratios': [3, 2]})

        # plot histogram on top subplot
        ax1.bar(range(num_clusters), dist)
        ax1.set_xticks(range(num_clusters))
        
        # remove all spines and ticks for bottom plot
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        # y lim would determine the number of images to display per cluster
        # a larger ylim will result in the images coinciding with each other
        ax2.set_ylim(5, 0)
        # align x axis with top plot
        ax2.set_xlim(ax1.get_xlim())

        for x, image_list in enumerate(cluster_images):
            for y, im in enumerate(image_list):
                im.thumbnail((60,60), Image.LANCZOS)
                # convert to RGBA to allow for transparency 
                im = im.convert('RGBA').rotate(45, expand=True)
                imagebox = OffsetImage(im)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                ax2.add_artist(ab)

        plt.savefig('shelf_images_hist/'+img.split('.')[0]+'.png')

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
    dist_plot()