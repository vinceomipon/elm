import cv2
import numpy as np
import random


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.ELMImage import *

class k_means_calc:
    # only pass centroids for argument purposes, to verify that implementation works
    def __init__(self, k: int, img: ELMImage, centroids=None):
        random.seed(42)
        self.k = k
        self.img = img
        self.r, self.g, self.b = self.img.flatten_channels(self.img.rgb_arr)
        self.pixels = len(self.r)
        if centroids is None:
            self.centroids = []
        else:
            self.centroids = centroids
    
    def k_means(self):
        # Convert 3D array [x, y, colour_channels] into flatten 1d array [[colour_channels]]
        # indice of array matches pixel (x,y)
        pixel_vals = self.img.bgr_arr.reshape((-1,3))
        pixel_vals = np.float32(pixel_vals)

        # max 100 iterations of k_means with accuraccy 85%
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)

        retval, labels, centers = cv2.kmeans(pixel_vals, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]

        segmented_image = segmented_data.reshape((self.img.bgr_arr.shape))

        plt.savefig(f'data/output/two_means/{self.k}k-means.png', dpi=300, transparent=True, bbox_inches='tight')


    






# Implement these later maybe

    
