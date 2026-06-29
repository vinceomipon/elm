import cv2
import numpy as np
import random
from pathlib import Path
from scipy.ndimage import binary_fill_holes


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
    
    def k_means(self, select: int):
        # Convert 3D array [x, y, colour_channels] into flatten 1d array [[colour_channels]]
        # indice of array matches pixel (x,y)
        # select = 0 is bgr, select = 1 is hsv
        if select == 0:
            pixel_vals = self.img.hsv_arr.reshape((-1,3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1)
            source_img = self.img.hsv_arr
        elif select == 1:
            pixel_vals = self.img.bgr_arr.reshape((-1,3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
            source_img = self.img.bgr_arr
            
        
        pixel_vals = np.float32(pixel_vals)


        retval, labels, centers = cv2.kmeans(pixel_vals, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # get the original dimensions of the img
        h, w, c = self.img.bgr_arr.shape

        cluster_imgs = []

        # the method caller should use a for loop to iterate over the number of k's used
        for i in range(self.k):
            # create a completely blank img with the shape of the original
            one_cluster_img = np.zeros((h * w, c), dtype=np.uint8)

            cluster_mask = (labels.flatten() == i)

            one_cluster_img[cluster_mask] = [255, 255, 255]

            one_cluster_img = one_cluster_img.reshape(self.img.bgr_arr.shape)

            kernel = np.ones((3,3), np.uint8)
            # remove any noise from the image

            b, g, r = one_cluster_img[2100, 1250]
            if b == 255 and g == 255 and r == 255:
                return one_cluster_img

            # cluster_imgs.append(one_cluster_img.reshape(self.img.bgr_arr.shape))

        # returns a list of array clusters
        return cluster_imgs

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
    
    def panel_mask(self, img: ELMImage, select: int):
        lower1 = np.array([0, 100, 50])
        upper1 = np.array([30, 255, 255])
        mask1 = cv2.inRange(img.hsv_arr, lower1, upper1)

        lower2 = np.array([160, 100, 50])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(img.hsv_arr, lower2, upper2)

        # merge the masks into one
        final_mask = cv2.bitwise_or(mask1, mask2)
        
        
        kernel = np.ones((3,3), np.uint8)
        out = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel=kernel, iterations=30)

        print("You may apply a fill of black or white, if panel not segmented properly")

        while True:
            print("please select coordinates")
            coords = self.get_rectangle_coords(out)


        

        

        return out
    
    def click_event(self, event, x, y, flags, coord_list):
        # If left click triggered save the coordinates
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # if first edge
            if len(coord_list) == 0:
                coord_list.append((round(x / 0.25), round(y / 0.25)))
                print(f"Corner 1, ({round(x / 0.25)}, {round(y / 0.25)})")
            
            elif len(coord_list) == 1:
                if coord_list[0][0] < x and coord_list[0][1] < y:
                    coord_list.append((round(x / 0.25), round(y / 0.25)))
                    print(f"Corner 2, ({round(x / 0.25)}, {round(y / 0.25)})")
                else:
                    print("Invalid corner, please choose (x,y) coords greater than the first edge")

    
    def get_rectangle_coords(self, mask) -> list:
        coords = []
        window_name = "Currnet Mask State"
        cv2.namedWindow(window_name)

        cv2.setMouseCallback(window_name, self.click_event, param=coords)

        max_clicks = 2

        scaled_down = cv2.resize(mask, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        
        
        while True:
            # display current state of mask
            
            cv2.imshow(window_name, scaled_down)

            key = cv2.waitKey(1) & 0xFF
            if len(coords) >= max_clicks:
                print("2 edges reached, closing window")
                break

            # if user clicks esc, close window
            if key == 27:
                print("Selection cancelled by user")
                break

        cv2.destroyAllWindows()
        return coords

    
    


    
    # def morph_select(self, mask, select: int):
    #     kernel = np.ones((3,3), np.uint8)
        

    #     if select == 4:
    #         y_min, y_max = 2800, 3300
    #         x_min, x_max = 1500, 1800

    #         roi = mask[y_min:y_max, x_min:x_max]
    #         fixed_roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=30)

    #         mask[y_min:y_max, x_min:x_max] = fixed_roi

    #         y_min, y_max = 2500, 2600
    #         x_min, x_max = 1100, 1300

    #         roi = mask[y_min:y_max, x_min:x_max]
    #         fixed_roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=40)

    #         mask[y_min:y_max, x_min:x_max] = fixed_roi
    #     if select == 3:
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    #     if select == 2:
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=20)


    #     return mask

    





    def save_segmentation(self, channel_arr, out_dir: Path, select: int):
        seg_performed = "healthy" if select == 0 else "panel" if select == 1 else ""

        out_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"{seg_performed}_seg_{self.img.dir.with_suffix(".png").name}"
        out_path = out_dir / f"{self.img.dir.parent.name}_{out_filename}"
        cv2.imwrite(str(out_path), channel_arr)

    
