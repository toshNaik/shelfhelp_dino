# Shelfhelp Dino
## The Problem
Accurate localization within grocery store environments poses a considerable challenge due to the predominantly featureless nature of the surroundings when analyzed in the context of Lidar and depth data. This led to the hypothesis that incorporating semantic information, derived from visual features extracted from images may result in improving the performance of localization in such environments.

This repository contains code that uses the DINO model by Meta, YOLOv5, and KMeans algorithm to represent grocery store images as a distribution of products. 

## Background and Objective
- This work is part of my independent study for Fall 2023. 
- The primary aim was to enhance the localization in a grocery store by incorporating distributions obtained from products placed between aisles.
- This involves embedding the features obtained from images on the map. This work assumes insignificant changes in product placement within the grocery store. 
- Figure 1a and 1b show a shelf with products and the corresponding distribution for that shelf.

<center>
<figure>
<img src="shelf_images/shelf-1.jpg" width="500">
<b><figcaption>Figure 1a: shelf image</figcaption></b>
</figure>
</center>

<center>
<figure>
<img src="shelf_images_hist/shelf-1.png" width="500">
<b><figcaption>Figure 1b: shelf distribution image</figcaption></b>
</figure>
</center>


## DINO
- DINO is a framework to train a self-supervised model to learn representations from images. It employs a dual-network architecture, consisting of a student and a teacher network. Figure 2 shows the architecture of the model.
- It employs a multi-crop strategy to train the model. Multiple views of the same image are created at a 'global' and 'local' scale and passed to the student and teacher network. The student network learns to capture the global view of the image from the local views.
- The parameters of the teacher network are updated using the exponential moving average of the student network.
- The loss is cross-entropy of the embeddings predicted by the student and the teacher network. This is used to update the parameters of the student network.

<center>
<figure>
<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*C1_JK8uZIg_zCapv9d0ITA.png" width="500">
<b><figcaption>Figure 2: DINO</figcaption></b>
</figure>
</center>

## Image Retrieval Task
- The DINO ViT model was finetuned on custom dataset obtained by scraping images from google.
- About **16000** images were obtained using queries from a FDC(FoodDataCentral) dataset.
- The model was then evaluated on the image retrieval task.
- The model was given a query image and the model was expected to retrieve the top-n most similar images from the dataset. There were 70 products and 350 target images (5 for each product).
- The model was evaluated on the mAP score. The mAP score is the mean of the average precision scores for each query image.

| - | Original  | Finetuned |
| ------------- | ------------- | ------------- |
| top-1 |  0.986  | 1  |
| top-3 | 0.99  | 1  |
| top-5 | 0.973 | 0.987 |

## Clustering
- To find the optimal number of clusters, the elbow method was used. However, the elbow method was not very useful in this case as the dataset was huge and the embeddings were very high dimensional. 
- However, for this use case where the clusters were used as a representation of the distribution of products on the shelf, and not for finding common features, I ended up using 30 clusters as it gave a good balance between the number of clusters and the number of products that would fall withiin each cluster in a grocery store.

## Testing the distribution
- The distributions were needed to see if the resampling stage of Monte Carlo Localization could be improved.
- To test the distributions, I created a map of shelves and manually placed product distributions on the map.
- I then used a few query images (different views or placements from the ones used to create the map) and tested if the model could correctly identify the position of the query image on the map (see figure 3a, 3b and 3c).

<center>
<figure>
<img src="grocery_map1.png" width="500">
<b><figcaption>Figure 3a: Map</figcaption></b>
</figure>
</center>

<center>
<figure>
<img src="shelf_query3.jpg" width="500">
<b><figcaption>Figure 3b: Query image</figcaption></b>
</figure>
</center>

<center>
<figure>
<img src="grocery_map2.png" width="500">
<b><figcaption>Figure 3c: Estimated shelf position on map</figcaption></b>
</figure>
</center>

## Other:
- I also made APIs for the models and other utility functions which weren't present in the original shelfhelp repository. These APIs can be used to refactor the code and make it more modular.
- At the start of the semester I also attempted to work on the motion model of the Monte Carlo Localization model but I was not able to make much progress on it. I started out by manually integrating the IMU data to obtain position and angles however, there were problems with drifting. I ended up using a library to get accurate angles but I was not able to get good enough position data.

## Future Work:
- Making the entire Monte Carlo Localization pipeline. Here I just focused on making the resampling stage better. The next step would be to integrate this with the motion model and the sensor model.
- Thoroughly testing the distribution matching. I only got to test it on a few shelf images. I would like to test it on more images (possibly in a real grocery store) and also try different similarity metrics. Functions for different similarity metrics have already been implemented in similarity.py.
- Completing the entire pipeline to publish it to RSS.