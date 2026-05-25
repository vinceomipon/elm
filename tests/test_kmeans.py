import cv2
import numpy as np


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.models import *

# tests the x,y coordinates generated from the function
def test_rgb_centroid_gen():
    k = 2
    centroids = generate_rgb_centroids(k)

    for centroid in centroids:
        for i in range(3):
            assert 0 <= centroid[i] and 255 >= centroid[i]

def test_rgb_centroid_gen():
    k = 2
    centroids = generate_rgb_centroids(k)

    for centroid in centroids:
        assert 0 <= centroid[0] and 255 >= centroid[0]
        assert 0 <= centroid[1] and 255 >= centroid[1]
        assert 0 <= centroid[2] and 255 >= centroid[2]

def test_kmeans():
    k = 2
    dir = "data/raw/20241116_111731.jpg"
    img = ELMImage(dir)

    k_means(img, k, 1)
    

    



    
