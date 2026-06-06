import cv2
import numpy as np
import random
from pathlib import Path


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
    
    def k_means(self, out_dir: Path, select: int, target_cluster: int):
        # Convert 3D array [x, y, colour_channels] into flatten 1d array [[colour_channels]]
        # indice of array matches pixel (x,y)
        # select = 0 is bgr, select = 1 is hsv
        if select == 0:
            pixel_vals = self.img.hsv_arr.reshape((-1,3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 2.0)
            source_img = self.img.hsv_arr
        elif select == 1:
            pixel_vals = self.img.bgr_arr.reshape((-1,3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
            source_img = self.img.bgr_arr
            
        
        pixel_vals = np.float32(pixel_vals)


        retval, labels, centers = cv2.kmeans(pixel_vals, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # get the original dimensions of the img
        h, w, c = self.img.bgr_arr.shape

        # the method caller should use a for loop to iterate over the number of k's used
        # create a completely blank img with the shape of the original
        one_cluster_img = np.zeros((h * w, c), dtype=np.uint8)


        cluster_mask = (labels.flatten() == target_cluster)

        one_cluster_img[cluster_mask] = [255, 255, 255]

        # returns a single cluster
        return one_cluster_img.reshape(self.img.bgr_arr.shape)

        # centers = np.uint8(centers)
        # segmented_data = centers[labels.flatten()]

        # segmented_image = None

        # # grayscale the segmented image
        # gray_segmented_image = None
        # if select == 0:
        #     segmented_image = segmented_data.reshape(self.img.hsv_arr.shape)
        #     gray_segmented_image = segmented_image
        # elif select == 1:
        #     segmented_image = segmented_data.reshape(self.img.bgr_arr.shape)
        #     gray_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)


        # self.save_k_means(gray_segmented_image, out_dir)

    def save_k_means(self, channel_arr, out_dir: Path, cluster: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"{self.k}k_cluster_{cluster}_{self.img.dir.with_suffix(".png").name}"
        out_path = out_dir / f"{self.img.dir.parent.name}_{out_filename}"
        cv2.imwrite(str(out_path), channel_arr)

    
