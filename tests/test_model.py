import cv2
import numpy as np
from pathlib import Path


from src.processing.ELMImage import ELMImage

# checks getter methods
def test_get_dir():
    dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6612.png"
    img = ELMImage(dir) 
    assert dir == img.dir

def test_get_bgr_arr():
    dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6612.png"
    img = ELMImage(dir)
    expected_bgr_arr = cv2.imread(dir)
    actual_bgr_arr = img.bgr_arr

    # checks if their dimensions are the same 
    assert img.bgr_arr.shape == expected_bgr_arr.shape

    # check if all pixel values are equal
    assert np.array_equal(actual_bgr_arr, expected_bgr_arr)

def test_get_hsv_arr():
    dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6612.png"
    img = ELMImage(dir)
    expected_img = cv2.imread(dir)
    expected_hsv_arr = cv2.cvtColor(expected_img, cv2.COLOR_BGR2HSV)
    actual_hsv_arr =  img.hsv_arr

    assert expected_hsv_arr.shape == actual_hsv_arr.shape

    assert np.array_equal(actual_hsv_arr, expected_hsv_arr)

def test_get_dimensions():
    dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6612.png"
    img = ELMImage(dir)
    expected_img = cv2.imread(dir)
    expected_dim = expected_img.shape
    actual_dim =  img.bgr_arr.shape
    assert expected_dim == actual_dim
    

# disregard test, just tryna understand how the rgb_arr is stored
def test_rgb_shape():
    dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6612.png"
    img = cv2.imread(dir)
    rgb_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(rgb_arr.shape)
    print(rgb_arr)

def test_rgb_scatter():
    dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6612.png"
    img = ELMImage(dir)
    img.disp_rgb_scatter()

def test_hsv_scatter():
    dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6612.png"
    img = ELMImage(dir)
    img.disp_hsv_scatter()

def test_resize_img():
    dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6612.png"
    img = ELMImage(dir)
    expected_img = cv2.imread(dir)
    

    height = 400
    width = 300

    # this is true since 'cv2.imread(dir)' outputs a bgr_arr
    expected_bgr_arr = expected_img
    expected_rgb_arr = cv2.cvtColor(expected_img, cv2.COLOR_BGR2RGB)
    expected_hsv_arr = cv2.cvtColor(expected_img, cv2.COLOR_BGR2HSV)

    # resize the image
    expected_bgr_arr = cv2.resize(expected_bgr_arr, (width, height))
    expected_rgb_arr = cv2.resize(expected_rgb_arr, (width, height))
    expected_hsv_arr = cv2.resize(expected_hsv_arr, (width, height))

    # check if resize was performed sucessfully
    assert img.resize_img(width, height)

    actual_rgb_arr = img.rgb_arr
    actual_bgr_arr = img.bgr_arr
    actual_hsv_arr = img.hsv_arr

    assert actual_rgb_arr.shape == expected_rgb_arr.shape
    assert actual_bgr_arr.shape == expected_bgr_arr.shape
    assert actual_hsv_arr.shape == expected_hsv_arr.shape



    assert np.array_equal(actual_rgb_arr, expected_rgb_arr)
    assert np.array_equal(actual_bgr_arr, expected_bgr_arr)
    assert np.array_equal(actual_hsv_arr, expected_hsv_arr)

def test_disp_hsv_img1():
    img_dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6613.png"
    out_dir = Path.cwd() / "data" / "processed" / "hsv_imgs" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)

    img.disp_hsv_img(out_dir)

def test_disp_hsv_img2():
    img_dir = Path.cwd() / "data" / "processed" / "sp2" / "IMG_6609.png"
    out_dir = Path.cwd() / "data" / "processed" / "hsv_imgs" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)

    img.disp_hsv_img(out_dir)

def test_disp_hsv_img3():
    img_dir = Path.cwd() / "data" / "processed" / "sp3" / "IMG_6674.png"
    out_dir = Path.cwd() / "data" / "processed" / "hsv_imgs" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)

    img.disp_hsv_img(out_dir)

def test_disp_hsv_img4():
    img_dir = Path.cwd() / "data" / "processed" / "sp4" / "IMG_6643.png"
    out_dir = Path.cwd() / "data" / "processed" / "hsv_imgs" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)

    img.disp_hsv_img(out_dir)

def test_disp_hsv_mono_img1():
    img_dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6613.png"
    out_dir = Path.cwd() / "data" / "processed" / "hsv_channels" / f"{img_dir.name}"
    img = ELMImage(img_dir)

    img.disp_hsv_mono_img(out_dir, select="h")
    img.disp_hsv_mono_img(out_dir, select="s")
    img.disp_hsv_mono_img(out_dir, select="v")



# checks that flatten_channels fnc actually flatten arr from nxn 1xn^2
def test_flatten_channels():
    dir = "data/test-images/red_2x2.png"
    img = ELMImage(dir)

    actual_channels = img.flatten_channels(img.rgb_arr)

    for colours in actual_channels:
        assert colours.size == 4



    
    




    
    
