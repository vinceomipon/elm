# import libraries
import cv2
import numpy as np

class ELMImage:

    """
    Initialize the ELMImage Object using reference to image

    Args: 
        image: the image to reference to, must have valid path to file

    """
    def __init__(self, image):
        self._image = image
        self._bgr_arr = cv2.imread(image)
        self._rgb_arr = cv2.cvtColor(self._bgr_arr, cv2.COLOR_BGR2RGB)
        self._hsv_arr = cv2.cvtColor(self._bgr_arr, cv2.COLOR_BGR2HSV)
        self._illuminated_mask = None
        self._panel_mask = None
    

    # Define get methods

    def get_image(self):
        return self._image
    
    def get_bgr_arr(self):
        return self._bgr_arr
    
    def get_hsv_arr(self):
        return self._hsv_arr

