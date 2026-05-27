import cv2
import numpy as np


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.models import *

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

def test_hsv_centroid_gen():
    k = 2
    dir = "data/raw/20241116_111731.jpg"
    two_k_means = k_means_calc(k, dir)
    two_k_means.generate_hsv_centroids()
    centroids = two_k_means.centroids

    for centroid in centroids:
        assert 0 <= centroid[0] and 179 >= centroid[0]
        assert 0 <= centroid[1] and 255 >= centroid[1]
        assert 0 <= centroid[2] and 255 >= centroid[2]

def test_rgb_centroid_gen():
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

    calc.k_means(1)

def test_distance():
    k = 2
    dir = "data/test-images/red_2x2.png"
    img =  ELMImage(dir)
    calc = k_means_calc(k, img)

    a1 = np.array([0, 0, 0])
    a2 = np.array([2, 2, 2])
    expected_dist = np.linalg.norm(a1 - a2)
    actual_dist = calc.distance(a1, a2)

    assert expected_dist

def test_calc_min_distance1():
    k = 3
    dir = "data/test-images/red_2x2.png"
    img = ELMImage(dir)
    centroids = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    calc = k_means_calc(k, img, centroids)
    r_flat = [0]
    g_flat = [255]
    b_flat = [0]
    i = 0
    expected_centroid = 1
    actual_centroid = calc.calc_min_distance(r_flat, g_flat, b_flat, i)

    assert actual_centroid == expected_centroid

def test_calc_min_distance2():
    k = 2
    dir = "data/test-images/red_2x2.png"
    centroids = [[0,0,0], [10,10,10]]
    img = ELMImage(dir)
    calc = k_means_calc(k, img, centroids)
    r_flat = [5]
    g_flat = [5]
    b_flat = [5]
    i = 0
    expected_centroid = 0
    actual_centroid = calc.calc_min_distance(r_flat, g_flat, b_flat, i)
    assert expected_centroid == actual_centroid
    


    



    
