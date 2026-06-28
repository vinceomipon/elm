import cv2
import numpy as np
import random
from pathlib import Path
from scipy.ndimage import binary_fill_holes


# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.ELMImage import *

def area_coverage(mask1, mask2) -> float:
    # count the number of white pixels in the panel segmentation
    actual_area = np.count_nonzero(mask2)

    healthy_area = np.count_nonzero(mask1)

    return (healthy_area / actual_area) * 100

