import cv2
import os
import numpy as np
from pathlib import Path



# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.ELMImage import *

# tests the x,y coordinates generated from the function
def test_constructor1():
    k = 2
    dir = "data/test-images/red_2x2.png"
    img = ELMImage(dir)
    calc = k_means_calc(k, img)

    # check if centroids list is empty if didnt pass value
    assert not calc.centroids
    assert calc.k == k

# test if centroids was passed
def test_constructor2():
    k = 3
    dir = "data/test-images/red_2x2.png"
    img = ELMImage(dir)
    expected_centroids = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    calc = k_means_calc(k, img, centroids=expected_centroids)
    actual_centroids = calc.centroids

    assert np.array_equal(expected_centroids, actual_centroids)

    k = 2
    dir = "data/raw/20241116_111731.jpg"
    two_k_means = k_means_calc(k, dir)
    two_k_means.generate_rgb_centroids()
    centroids = two_k_means.centroids

    for centroid in centroids:
        assert 0 <= centroid[0] and 255 >= centroid[0]
        assert 0 <= centroid[1] and 255 >= centroid[1]
        assert 0 <= centroid[2] and 255 >= centroid[2]

def test_twokmeans1():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6613.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "2k" / f"{img_dir.parent.name}"  
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_twokmeans2():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp2" / "IMG_6609.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "2k" / f"{img_dir.parent.name}" 
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_twokmeans3():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp3" / "IMG_6674.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "2k" / f"{img_dir.parent.name}" 
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_twokmeans4():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp4" / "IMG_6643.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "2k" / f"{img_dir.parent.name}" 
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_twokmeans5():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp2" / "IMG_6599.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "2k" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_twokmeans6():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6596.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "2k" / f"{img_dir.parent.name}" 
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_twokmeans7():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp3" / "IMG_6626.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "2k" / f"{img_dir.parent.name}" 
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_twokmeans8():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp4" / "IMG_6639.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "2k" / f"{img_dir.parent.name}" 
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)



def test_threekmeans1():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6613.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}" 
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)


def test_threekmeans2():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp2" / "IMG_6609.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}" 
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)
    

def test_threekmeans3():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp3" / "IMG_6674.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_threekmeans4():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp4" / "IMG_6643.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_threekmeans5():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp2" / "IMG_6599.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_threekmeans6():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6596.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_threekmeans7():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp3" / "IMG_6626.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_threekmeans8():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp4" / "IMG_6639.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_threekmeans9():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp3" / "IMG_6622.png"
    out_dir = Path.cwd() / "data" / "output" / "k_means" / "3k" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    centroid_imgs = calc.k_means(out_dir, select=1)

    for i in range(k):
        
        calc.save_k_means(centroid_imgs[i], out_dir, i)

def test_panel_mask():
    k = 3
    img_dir = Path.cwd() / "data" / "processed" / "sp4" / "IMG_6639.png"
    out_dir = Path.cwd() / "data" / "output" / "panel_mask" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    segmented_img = calc.panel_mask(img)

    calc.save_k_means(segmented_img, out_dir, 0)

def test_panel_mask2():
    k = 2
    img_dir = Path.cwd() / "data" / "raw" / "old" / "20241116_111731.jpg"
    out_dir = Path.cwd() / "data" / "output" / "panel_mask" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    segmented_img = calc.panel_mask(img)

    calc.save_k_means(segmented_img, out_dir, 0)

def test_panel_mask3():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp4" / "IMG_6637.png"
    out_dir = Path.cwd() / "data" / "output" / "panel_mask" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    segmented_img = calc.panel_mask(img)

    calc.save_k_means(segmented_img, out_dir, 0)

def test_panel_mask4():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp3" / "IMG_6626.png"
    out_dir = Path.cwd() / "data" / "output" / "panel_mask" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    segmented_img = calc.panel_mask(img, select=3)

    calc.save_k_means(segmented_img, out_dir, 0)

def test_panel_mask5():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp2" / "IMG_6609.png"
    out_dir = Path.cwd() / "data" / "output" / "panel_mask" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    segmented_img = calc.panel_mask(img, select=2)

    calc.save_k_means(segmented_img, out_dir, 0)

def test_panel_mask6():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp3" / "IMG_6622.png"
    out_dir = Path.cwd() / "data" / "output" / "panel_mask" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    segmented_img = calc.panel_mask(img, select = 3)

    calc.save_k_means(segmented_img, out_dir, 0)

def test_panel_mask6():
    k = 2
    img_dir = Path.cwd() / "data" / "processed" / "sp1" / "IMG_6596.png"
    out_dir = Path.cwd() / "data" / "output" / "panel_mask" / f"{img_dir.parent.name}"
    img = ELMImage(img_dir)
    calc = k_means_calc(k, img)
    segmented_img = calc.panel_mask(img, select = 1)

    calc.save_k_means(segmented_img, out_dir, 0)



# # test print function of python
# def test_pathlib(capsys):
#     with capsys.disabled():
#     #     for p in Path().iterdir():
#     #         print(p)

#         my_dir = Path("data")
#         my_file = Path("IMG_6592.HEIC")


#         print(my_dir.parent.parent)
#         print(my_dir.absolute().parent)
#         print(my_dir.exists())
#         print(Path("..").resolve())
#         print(Path(__file__).resolve().parent)
#         print(my_file.absolute())


    




    



    
