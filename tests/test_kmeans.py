import cv2
import numpy as np


# imports all functions made in kmeans
from src.core.kmeans import *

# tests the x,y coordinates generated from the function
def test_center_gen():
    dir = "data/raw/20241116_110324.jpg"
    img = cv2.imread(dir)
    rgb_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    k_means(rgb_arr, 2)
    
