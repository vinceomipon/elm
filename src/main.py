import cv2
import os
import numpy as np
from pathlib import Path



# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.ELMImage import *
from src.metrics.area_calculator import *

def main():
    another = True
    while another:
        file_path = get_file_path()

        print("Performing healthy segmentation of solar panel")
        k = 2
        img = ELMImage(file_path)
        calc = k_means_calc(k, img)
        out_dir = Path.cwd() / f"data/output/healthy_segmentation/{img.dir.parent.name}"

        healthy_segment = calc.k_means(select=1)

        print(f"Healthy segmentation performed, saving it in {out_dir}")
        img_dir = calc.save_segmentation(healthy_segment, out_dir, select=0)
        healthy_img = ELMImage(img_dir)
        print("Healthy segmentation saved")

        print("Performing panel segmentation of solar panel")
        panel_segmentation = calc.panel_mask(img, select=1)

        out_dir = Path.cwd() / f"data/output/panel_segmentation/{img.dir.parent.name}"
        print(f"Panel segmentation performed, saving it in {out_dir}")
        img_dir = calc.save_segmentation(panel_segmentation, out_dir, select=1)
        panel_img = ELMImage(img_dir)
        print("Panel segmentation saved")

        
        print("Performing ELM Coverage calculation")
        elm_coverage = area_coverage(healthy_img.bgr_arr, panel_img.bgr_arr)
        print(f"ELM Coverage {elm_coverage:.2f}")

        user_input = input("Do you want to perform another operation? (yes/no): ").strip().lower()
        another = user_input in ("yes", "y", "true")
    
    print("Done ELM testing")





    



    

def get_file_path() -> Path:
    # get the input img file from user
    file_path = ""

    while True:
        user_string = input("Enter a valid image file located in data/processed/sp*: ").strip("'\"")
        file_path = Path.cwd() / "data/processed" / user_string

        if file_path.is_file():
            print("File found")
            break

        print("Invalid file try again")
    
    return file_path


if __name__ == "__main__":
    main()