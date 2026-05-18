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
    
    # display scatter plots
    def disp_rgb_scatter(self):
        # create 3d plot and figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # the channels are stored as 2D arrays, must flatten them to 1d for scatter to work
        r, g, b = cv2.split(self.rgb_arr)

        # converts 2D to 1D arr
        r_flat = r.flatten()
        g_flat = g.flatten()
        b_flat = b.flatten()

        # create color map
        rgb_colors = np.vstack((r_flat, g_flat, b_flat)).T / 255.0

        ax.scatter(r_flat, g_flat, b_flat, c=rgb_colors, marker='.')

        ax.set_xlabel('R axis')
        ax.set_ylabel('G axis')
        ax.set_zlabel('B axis')

        plt.savefig("data/output/rgb_scatter.png", dpi=300, bbox_inches='tight')

        plt.close(fig)
    
    
    # Permanently resizes the image
    # returns True if succesfully completed
    def resize_img(self, width: int, height: int):
        self.bgr_arr = cv2.resize(self.bgr_arr, (width, height))
        self.rgb_arr = cv2.resize(self.rgb_arr, (width, height))
        self.hsv_arr = cv2.resize(self.hsv_arr, (width, height))
        return True


