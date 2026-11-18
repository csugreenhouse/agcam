import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

graph_util = importlib.import_module("utils.graph_util")
openCV_util = importlib.import_module("utils.openCV_util")

plant_finder = importlib.import_module("plant_finders.abby_plant_finder")

def test_accuracy(test_src_path, key_path):
    test_src_path = Path(test_src_path)
    key_path = Path(key_path)

    answer_key_mask = openCV_util.select_white_pixels(cv2.imread(str(key_path)))
    answer_key_contours = openCV_util.get_contours_from_mask(answer_key_mask)
    graph_util.plot_contour_and_mask(key_path, answer_key_contours, answer_key_mask) #optional visual sanity check

    test_mask = plant_finder.find_plants(test_src_path)
    test_contours = openCV_util.get_contours_from_mask(test_mask)
    graph_util.plot_contour_and_mask(test_src_path, test_contours, test_mask) #optional visual sanity check

    accuracy = openCV_util.compare_two_masks(test_mask, answer_key_mask) 

    print(f"Testing Complete!")

#if __name__ == "__main__":

test_accuracy("tests/testimages/plant_detect_accuracy_test/IMG_3397.jpg", "tests/testimages/plant_detect_accuracy_test/IMG_3397_key.jpg")

test_accuracy("tests/testimages/plant_detect_accuracy_test/force_50_accuracy_test.jpg", "tests/testimages/plant_detect_accuracy_test/force_key.jpg")

test_accuracy("tests/testimages/plant_detect_accuracy_test/force_25_accuracy_test.jpg", "tests/testimages/plant_detect_accuracy_test/force_key.jpg")