import cv2
import numpy as np


from src.processing.models import ELMImage

# check if loader passed successfully
def test_get_image():
    dir = "data/raw/20241116_110324.jpg"
    img = ELMImage("data/raw/20241116_110324.jpg")
    assert dir == img.get_image()

def test_get_bgr_arr():
    dir = "data/raw/20241116_110324.jpg"
    image = cv2.imread(dir)
    cv2.imshow("Window", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
