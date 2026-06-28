import cv2
import os
import numpy as np
from pathlib import Path



# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.ELMImage import *
from src.metrics.area_calculator import *

def main():

    file_path = get_file_path()

    print("Performing healthy segmentation of solar panel")
    k = 2
    img = ELMImage(file_path)
    calc = k_means_calc(k, img)
    



    

def get_file_path() -> Path:
    # get the input img file from user
    file_path = ""

    while True:
        user_string = input("Enter a valid image file located in data/processed/sp*: ").strip("'\"")
        file_path = Path(user_string)

        if file_path.is_file():
            print("File found")
            break

        print("Invalid file try again")
    
    return file_path


if __name__ == "__main__":
    main()