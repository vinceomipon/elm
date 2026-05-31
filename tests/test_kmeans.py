import cv2
import numpy as np


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.ELMImage import *

# tests the x,y coordinates generated from the function
def test_constructor1():
    k = 2
    dir = "data/test-images/red_2x2.png"
    img = ELMImage(dir)
    calc = k_means_calc(k, img)

    # check if centroids list is empty if didnt pass value
    assert not calc.centroids
    assert calc.k == k

# test if centroids was passed
def test_constructor2():
    k = 3
    dir = "data/test-images/red_2x2.png"
    img = ELMImage(dir)
    expected_centroids = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    calc = k_means_calc(k, img, centroids=expected_centroids)
    actual_centroids = calc.centroids

    assert np.array_equal(expected_centroids, actual_centroids)

    k = 2
    dir = "data/raw/20241116_111731.jpg"
    two_k_means = k_means_calc(k, dir)
    two_k_means.generate_rgb_centroids()
    centroids = two_k_means.centroids

    for centroid in centroids:
        assert 0 <= centroid[0] and 255 >= centroid[0]
        assert 0 <= centroid[1] and 255 >= centroid[1]
        assert 0 <= centroid[2] and 255 >= centroid[2]

def test_kmeans():
    k = 2
    dir = "data/raw/20241116_111731.jpg"
    img = ELMImage(dir)
    calc = k_means_calc(k, img)

    calc.k_means()


    



    
