import cv2
import numpy as np


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.models import *

# tests the x,y coordinates generated from the function
def test_rgb_centroid_gen():
    k = 2
    dir = "data/raw/20241116_111731.jpg"
    two_k_means = k_means_calc(k, dir)
    centroids = two_k_means.generate_rgb_centroids()

    for centroid in centroids:
        for i in range(3):
            assert 0 <= centroid[i] and 255 >= centroid[i]

def test_rgb_centroid_gen():
    k = 2
    dir = "data/raw/20241116_111731.jpg"
    two_k_means = k_means_calc(k, dir)
    centroids = two_k_means.generate_rgb_centroids()

    for centroid in centroids:
        assert 0 <= centroid[0] and 255 >= centroid[0]
        assert 0 <= centroid[1] and 255 >= centroid[1]
        assert 0 <= centroid[2] and 255 >= centroid[2]

def test_kmeans():
    k = 2
    dir = "data/raw/20241116_111731.jpg"
    img = ELMImage(dir)
    calc = k_means_calc(k, img)

    calc.k_means(1)
    

    



    
