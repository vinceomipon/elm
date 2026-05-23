import cv2
import numpy as np
import random


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.models import *

def two_means(img: ELMImage, k: int):
    hsv_arr = img.hsv_arr
    rgb_arr = img.rgb_arr

    # dictionary for all clusters
    clusters = {}

    # idea for centroid key is the # of the centroid, value coresponds to the coordiante
    centroids = generate_centroids(img.width, img.height, k)

    # # iterate over entire image
    # for i in range(img.width):
    #     for j in range(img.height):

def generate_centroids(width: int, height: int, k: int):
    centroids = {}

    for i in range(k):
        # assign randomly generated centroids
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        centroids[f"c{i + 1}"] = (x,y)
    
    return centroids


# Calculates the Euclidean distance between the two based on the
# distance
def distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))






