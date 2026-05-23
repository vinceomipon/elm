import cv2
import numpy as np


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.models import *

# tests the x,y coordinates generated from the function
def test_centroid_gen_key():
    dir = "data/test-images/img1.png"
    img = ELMImage(dir)
    k = 2
    actual_centroids = generate_centroids(img.width, img.height, k)
    centroid_keys = list(actual_centroids.keys())
    for i in range(k):
        assert centroid_keys[i] == f"c{i + 1}"

# disregard test, want to see what it looks like in 2d
def test_centroid_plot():
    dir = "data/raw/20241116_110324.jpg"
    img = ELMImage(dir)

    # change this for the number of centroids
    k = 4

    centroids = generate_centroids(img.width, img.height, k)
    centroid_keys = list(centroids.keys())
    centroid_x = []
    centroid_y = []


    for i in range(len(centroid_keys)):
        # store the x,y value of the centroids
        centroid_x.append(centroids[f"c{i + 1}"][0]) 
        centroid_y.append(centroids[f"c{i + 1}"][1])

    plt.scatter(centroid_x, centroid_y, color='red', marker='.')
    plt.grid(True)
    plt.title('centroid plot')
    plt.show()
    plt.savefig("data/output/centroid_plot.png")



    
