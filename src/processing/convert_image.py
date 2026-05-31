import cv2
import numpy as np
import pillow_heif
from pathlib import Path

# Raw images are a .heic file convert them to png (i think?)

def heic_to_bgr(img_file: Path):
    # if the file doesnt exist
    if not img_file.is_file():
        raise FileNotFoundError("Image does not exist")
    
    heif_file = pillow_heif.open_heif(img_file, convert_hdr_to_8bit=True, bgr_mode=True)

    # convert buffer into NumPy array (notably a bgr_array)
    img = np.array(heif_file)
    
    return img

def save_img(img_file: Path, out_dir: Path):
    # convert img from heic to bgr
    img_arr = heic_to_bgr(img_file)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_file.with_suffix(".png").name
    # write the bgr_arr to specified directory
    cv2.imwrite(str(out_path), img_arr)


