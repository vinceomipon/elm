import cv2
import numpy as np
import pillow_heif
from pathlib import Path

# Raw images are a .heic file convert them to png (i think?)

def heif_to_arr():
    # Get the directory of where the raw files are stored
    image_path = Path.cwd() / "raw"

    heif_file = pillow_heif.open_heif(image_path / "")
