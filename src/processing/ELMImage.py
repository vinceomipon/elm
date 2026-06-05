# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

class ELMImage:

    """
    Initialize the ELMImage Object using reference to image

    Args: 
        image: the image to reference to, must have valid path to file

    """
    def __init__(self, dir: Path):
        self.dir = dir
        self.bgr_arr = cv2.imread(dir)
        self.rgb_arr = cv2.cvtColor(self.bgr_arr, cv2.COLOR_BGR2RGB)
        self.hsv_arr = cv2.cvtColor(self.bgr_arr, cv2.COLOR_BGR2HSV)
        self.height, self.width = self.rgb_arr.shape[:2]
        self.illuminated_mask = None
        self.panel_mask = None


    # display scatter plots
    def disp_rgb_scatter(self):
        # create 3d plot and figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        sampling_step = 5
        downsampled_img = self.rgb_arr[::sampling_step, ::sampling_step]

        # the channels are stored as 2D arrays, must flatten them to 1d for scatter to work
        r, g, b = cv2.split(downsampled_img)

        # converts 2D to 1D arr
        r_flat = r.flatten()
        g_flat = g.flatten()
        b_flat = b.flatten()

        # create color map
        rgb_colors = np.vstack((r_flat, g_flat, b_flat)).T / 255.0

        ax.scatter(r_flat, g_flat, b_flat, c=rgb_colors, marker='.', s=1)

        ax.set_xlabel('R axis')
        ax.set_ylabel('G axis')
        ax.set_zlabel('B axis')

        plt.savefig("data/output/rgb_scatter.png", dpi=300, bbox_inches='tight')

        plt.close(fig)
    
    def disp_hsv_scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')


        h, s, v = cv2.split(self.hsv_arr)
        
        h_flat = h.flatten()
        s_flat = s.flatten()
        v_flat = v.flatten()

        hsv_colors = np.vstack((h_flat, s_flat, v_flat)).T.astype(np.float32)

        norm_factors = np.array([179.0, 255.0, 255.0])

        normalized_hsv = hsv_colors / norm_factors

        # convert to rgb to colour the scatter plot
        rgb_colors = mcolors.hsv_to_rgb(normalized_hsv)

        ax.scatter(h_flat, s_flat, v_flat, c=rgb_colors, marker='.')

        ax.set_xlabel('H axis')
        ax.set_ylabel('S axis')
        ax.set_zlabel('V axis')

        plt.savefig("data/output/hsv_scatter.png", dpi=300, bbox_inches='tight')

        plt.close(fig)
    
    def disp_hsv_mono_img(self, out_dir: Path, select: str):
        # get the mono arr from hsv based on select
        mono_arr = self.hsv_mono_arr(select)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"{select}_{self.dir.with_suffix(".png").name}"
        out_path = out_dir / f"{self.dir.parent.name}_{out_filename}"
        cv2.imwrite(str(out_path), mono_arr)
    
    def disp_hsv_img(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"hsv_{self.dir.with_suffix(".png").name}"
        out_path = out_dir / f"{self.dir.parent.name}_{out_filename}"
        cv2.imwrite(str(out_path), self.hsv_arr)


    def hsv_mono_arr(self, select: str) -> cv2.typing.MatLike:
        h, s, v = cv2.split(self.hsv_arr)
        
        channel_map = {"h": h, "s": s, "v": v}
        mono_arr = channel_map[select]
        
        return mono_arr
    
    
    # Permanently resizes the image
    # returns True if succesfully completed
    def resize_img(self, width: int, height: int) -> bool:
        self.bgr_arr = cv2.resize(self.bgr_arr, (width, height))
        self.rgb_arr = cv2.resize(self.rgb_arr, (width, height))
        self.hsv_arr = cv2.resize(self.hsv_arr, (width, height))
        return True
    
    # splits the colour channels and flattens them
    def flatten_channels(self, channel_arr):
        a, b, c = cv2.split(channel_arr)
        a = np.array(a.flatten())
        b = np.array(b.flatten())
        c = np.array(c.flatten())
        return [a, b, c]

        


