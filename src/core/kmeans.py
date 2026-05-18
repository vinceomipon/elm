import cv2
import numpy as np


# imports all functions made in kmeans
from src.core.kmeans import *

def k_means(rgb_arr, k):
    clusters = {}

    # Ensures reporducibility by fixing the seed
    np.random.seed(23)

    for i in range(k):

        # picks a random integer x,y value from 0 to width of image
        x_rand = np.random.randint(0, rgb_arr.shape[1])
        y_rand = np.random.randint(0, rgb_arr.shape[0])
        print(x_rand)
        print(y_rand)

