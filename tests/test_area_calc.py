import cv2
import os
import numpy as np
from pathlib import Path



# imports all functions made in kmeans
from src.core.kmeans import *
from src.processing.ELMImage import *
from src.metrics.area_calculator import *

def test_area_calc1():
    healthy_segment_dir = Path(r"/home/vincesid/ORBIT/elm/data/output/k_means/2k/sp1/sp1_2k_cluster_0_IMG_6613.png")
    panel_segment_dir = Path(r"/home/vincesid/ORBIT/elm/data/output/panel_mask/sp1/sp1_2k_cluster_0_IMG_6613.png")
    healthy_mask = ELMImage(healthy_segment_dir)
    panel_mask = ELMImage(panel_segment_dir)

    coverage = area_coverage(healthy_mask.bgr_arr, panel_mask.bgr_arr)

    print(f"ELM Coverage {coverage}")

def test_area_calc2():
    healthy_segment_dir = Path(r"/home/vincesid/ORBIT/elm/data/output/k_means/2k/sp2/sp2_2k_cluster_1_IMG_6609.png")
    panel_segment_dir = Path(r"/home/vincesid/ORBIT/elm/data/output/panel_mask/sp1/sp1_2k_cluster_0_IMG_6613.png")
    healthy_mask = ELMImage(healthy_segment_dir)
    panel_mask = ELMImage(panel_segment_dir)

    coverage = area_coverage(healthy_mask.bgr_arr, panel_mask.bgr_arr)

    print(f"ELM Coverage {coverage}")