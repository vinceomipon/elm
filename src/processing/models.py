# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ELMImage:

    """
    Initialize the ELMImage Object using reference to image

    Args: 
        image: the image to reference to, must have valid path to file

    """
    def __init__(self, dir: str):
        self.dir = dir
        self.bgr_arr = cv2.imread(dir)
        self.rgb_arr = cv2.cvtColor(self.bgr_arr, cv2.COLOR_BGR2RGB)
        self.hsv_arr = cv2.cvtColor(self.bgr_arr, cv2.COLOR_BGR2HSV)
        self.illuminated_mask = None
        self.panel_mask = None
    

    # Define get methods

    def get_directory(self):
        return self.dir
    
    def get_bgr_arr(self):
        return self.bgr_arr
    
    def get_hsv_arr(self):
        return self.hsv_arr
    
    def get_rgb_arr(self):
        return self.rgb_arr
    
    # display scatter plots
    def disp_rgb_scatter(self):
        # create 3d plot and figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')


        r, g, b = cv2.split(self.rgb_arr)
        cv2.imshow('Blue Channel', b)
        cv2.waitKey(0)
    
    
    # Permanently resizes the image
    # returns True if succesfully completed
    def resize_img(self, width: int, height: int):
        self._bgr_arr = cv2.resize(self.bgr_arr, (width, height))
        self._rgb_arr = cv2.resize(self.rgb_arr, (width, height))
        self._hsv_arr = cv2.resize(self.hsv_arr, (width, height))
        return True


