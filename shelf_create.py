import os, cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from model_utils import Dino, yolo_load, yolo_forward

def create_map_grid():
    '''
    Creates a 100x100 grid map of the grocery store with 2 shelves
    '''
    map_grid = np.ones((100,100))
    map_grid[20:80, 20:40] = 0 # shelf 1
    map_grid[20:80, 60:80] = 0 # shelf 2
    return map_grid

def img_to_dist(img_path):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # run yolo to get patches
    crops = yolo_forward(img_to_dist.yolo_model, im)
    # get features
    features = img_to_dist.dino.extract_features(crops)
    if features.size == 0:
        return
    # cluster all features and get distribution as x*c1 + y*c2 + z*c3 + ... where c1 is cluster 1 center
    clusters = img_to_dist.kmeans.predict(features)
    dist = np.bincount(clusters, minlength=img_to_dist.num_clusters)
    return dist
img_to_dist.dino = Dino()
img_to_dist.yolo_model = yolo_load()
img_to_dist.num_clusters = 30
img_to_dist.kmeans = pickle.load(open('kmeans_models/kmeans_fitted30.pkl', 'rb'))

def embed_features():
    '''
    Embeds features into the map_grid
    '''
    dist1 = img_to_dist('shelf_images/shelf-1.jpg')
    dist12 = img_to_dist('shelf_images/shelf-1-2.jpg')
    dist2 = img_to_dist('shelf_images/shelf-2.jpg')
    dist23 = img_to_dist('shelf_images/shelf-2-3.jpg')
    dist3 = img_to_dist('shelf_images/shelf-3.jpg')
    dist4 = img_to_dist('shelf_images/shelf-4.jpg')
    dist45 = img_to_dist('shelf_images/shelf-4-5.jpg')
    dist5 = img_to_dist('shelf_images/shelf-5.jpg')
    dist56 = img_to_dist('shelf_images/shelf-5-6.jpg')
    # dist6 = img_to_dist('shelf_images/shelf-6.jpg')
    dist7 = img_to_dist('shelf_images/shelf-7.jpg')
    dist78 = img_to_dist('shelf_images/shelf-7-8.jpg')
    dist8 = img_to_dist('shelf_images/shelf-8.jpg')
    dist89 = img_to_dist('shelf_images/shelf-8-9.jpg')
    dist9 = img_to_dist('shelf_images/shelf-9.jpg')
    
    # create dictionary of distributions to pixel locations ((upperleft)(lowerright))
    dist_to_location = dict()
    dist_to_location[tuple(dist1)] = ((35,60),(45,80))
    dist_to_location[tuple(dist12)] = ((35,50),(45,70))
    dist_to_location[tuple(dist2)] = ((35,40),(45,60))
    dist_to_location[tuple(dist23)] = ((35,30),(45,50))
    dist_to_location[tuple(dist3)] = ((35,20),(45,40))
    dist_to_location[tuple(dist4)] = ((55,20),(65,40))
    dist_to_location[tuple(dist45)] = ((55,30),(65,50))
    dist_to_location[tuple(dist5)] = ((55,40),(65,60))
    dist_to_location[tuple(dist56)] = ((55,50),(65,70))
    # dist_to_location[tuple(dist6)] = ((55,60),(65,80))
    dist_to_location[tuple(dist7)] = ((75,60),(85,80))
    dist_to_location[tuple(dist78)] = ((75,50),(85,70))
    dist_to_location[tuple(dist8)] = ((75,40),(85,60))
    dist_to_location[tuple(dist89)] = ((75,30),(85,50))
    dist_to_location[tuple(dist9)] = ((75,20),(85,40))
    return dist_to_location


if __name__ == '__main__':
    map_grid = create_map_grid()
    dist_to_location = embed_features()
    fig, ax = plt.subplots()
    plt.imshow(map_grid, cmap='gray')
    plt.title('Grocery Map')
    plt.savefig('grocery_map1.png')


    query_dist = img_to_dist('shelf_query3.jpg')
    # find closest distribution
    closest_dist = None
    min_distance = float('inf')
    for dist in dist_to_location:
        # TODO: try different distance metrics
        distance = np.linalg.norm(np.array(dist)-np.array(query_dist))
        if distance < min_distance:
            min_distance = distance
            closest_dist = dist

    location = dist_to_location[closest_dist]
    rect = plt.Rectangle(location[0], 10, 20, facecolor='none', edgecolor='red')
    ax.add_patch(rect)
    plt.savefig('grocery_map2.png')


# drag_start = []
# def onpress(event):
#     drag_start.append((int(event.xdata), int(event.ydata)))

# drag_end = []
# def onrelease(event):
#     drag_end.append((int(event.xdata), int(event.ydata)))

# cidpress = plt.gcf().canvas.mpl_connect('button_press_event', onpress)
# cidrelease = plt.gcf().canvas.mpl_connect('button_release_event', onrelease)
# plt.show()

# xs = []
# ys = []
# for s, e in zip(drag_start, drag_end):
#     minx = min(s[0], e[0]) 
#     maxx = max(s[0], e[0])
#     miny = min(s[1], e[1])
#     maxy = max(s[1], e[1])
#     xs.append((minx, maxx))
#     ys.append((miny, maxy))



# fig, ax = plt.subplots()
# ax.imshow(map_grid, cmap='gray')
# for (minx, maxx), (miny, maxy) in zip(xs, ys):
#     rect = plt.Rectangle((minx, miny), maxx - minx, maxy - miny, 
#                         facecolor='none', edgecolor='red')
#     ax.add_patch(rect)
# plt.show()