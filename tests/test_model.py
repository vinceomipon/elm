import cv2
import numpy as np


from src.processing.models import ELMImage

# checks getter methods
def test_get_dir():
    dir = "data/raw/20241116_110324.jpg"
    img = ELMImage("data/raw/20241116_110324.jpg")
    assert dir == img.get_directory()

def test_get_bgr_arr():
    dir = "data/raw/20241116_110324.jpg"
    img = ELMImage(dir)
    expected_bgr_arr = cv2.imread(dir)
    actual_bgr_arr = img.get_bgr_arr()

    # checks if their dimensions are the same 
    assert img.get_bgr_arr().shape == expected_bgr_arr.shape

    # check if all pixel values are equal
    assert np.array_equal(actual_bgr_arr, expected_bgr_arr)

def test_get_hsv_arr():
    dir = "data/raw/20241116_110324.jpg"
    img = ELMImage(dir)
    expected_img = cv2.imread(dir)
    expected_hsv_arr = cv2.cvtColor(expected_img, cv2.COLOR_BGR2HSV)
    actual_hsv_arr =  img.get_hsv_arr()

    assert expected_hsv_arr.shape == actual_hsv_arr.shape

    assert np.array_equal(actual_hsv_arr, expected_hsv_arr)

# disregard test, just tryna understand how the rgb_arr is stored
def test_rgb_shape():
    dir = "data/raw/20241116_110324.jpg"
    img = cv2.imread(dir)
    rgb_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(rgb_arr.shape)
    print(rgb_arr)

def test_resize_img():
    dir = "data/raw/20241116_110324.jpg"
    img = ELMImage(dir)
    expected_img = cv2.imread(dir)
    

    height = 400
    width = 300

    # this is true since 'cv2.imread(dir)' outputs a bgr_arr
    expected_bgr_arr = expected_img
    expected_hsv_arr = cv2.cvtColor(expected_img, cv2.COLOR_BGR2HSV)

    # resize the image
    expected_bgr_arr = cv2.resize(expected_bgr_arr, (width, height))
    expected_hsv_arr = cv2.resize(expected_hsv_arr, (width, height))

    # check if resize was performed sucessfully
    assert img.resize_img(width, height)

    # actual_rgb_arr = img._rgb_arr






    
    
