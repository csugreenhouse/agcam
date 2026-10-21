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
reference_util = importlib.import_module("utils.reference_tag_util")
openCV_util = importlib.import_module("utils.openCV_util")

plant_finder = importlib.import_module("plant_finders.legacyHSV_plant_finder")

def plant_finder_test_functionality(src_path):
    src_path = Path(src_path)

    img = cv2.imread(str(src_path))

    if img is None:
        raise FileNotFoundError(f"Missing {src_path}.")
    
    mask = plant_finder.find_plants(src_path)
    contours = openCV_util.get_contours_from_mask(mask)

    graph_util.plot_contour_and_mask(src_path, contours, mask)

def test_select_white_pixels(src_path):
    src_path = Path(src_path)
    
    img = cv2.imread(str(src_path))

    if img is None:
        raise FileNotFoundError(f"Missing {src_path}.")

    mask = openCV_util.select_white_pixels(img)
    contours = openCV_util.get_contours_from_mask(mask)

    graph_util.plot_contour_and_mask(src_path, contours, mask, True, True)
    

if __name__ == "__main__":
    plant_finder_test_functionality("tests/testimages/IMG_3387.jpg")

    plant_finder_test_functionality("tests/testimages/IMG_3398.jpg")

    test_select_white_pixels("tests/testimages/plant_detect_accuracy_test/IMG_3397_key.jpg")