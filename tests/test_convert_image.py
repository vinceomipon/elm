import cv2
import os
import numpy as np
from pathlib import Path



# imports all functions made in kmeans
from src.processing.convert_image import *

# Only run this once probably
def test_heic_to_bgr():
    # get the parent directory
    # since there are sp1, sp2, etc. within raw use this as baseline?
    parent_dir = Path.cwd() / "data" / "raw"

    # create iterator for subdirectories existing within raw
    subdir_iterator = (entry for entry in parent_dir.iterdir() if entry.is_dir())

    # iterator for folders within raw
    for folder in subdir_iterator:

        out_dir = Path.cwd() / "data" / "processed" / folder.name
        # Iterate through all the .HEIC images and convert them to bgr
        for img_path in list(folder.glob("*.HEIC")) + list(folder.glob("*.heic")):
            save_img(img_path, out_dir=out_dir)

