import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plant_finder = importlib.import_module("plant_finders.legacyHSV_plant_finder")
graph_util = importlib.import_module("utils.graph_util")
du = importlib.import_module("utils.detect_util")


################ NOT FUNCTIONAL RN!

def process_image_key (src_path):
    src_path = Path(src_path)
    dst_path = Path(str(src_path).replace(".jpg", "_out.jpg"))

    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Src path missing {src_path}.")

    black_pixels = du.select_black_pixels(img)

    print(f"Found {len(black_pixels)} black pixel blobs in {src_path}.")

    graph_util.plot_mask(img, black_pixels, dst_path)
    print(f"blob test saved to: {dst_path}")

def test_accuracy(src_path, key_path):
    src_path = Path(src_path)
    key_path = Path(key_path)

    img = cv2.imread(str(src_path))

    #run the b/w on the key to get the answer key mask

    answer_key_mask = du.select_black_pixels(cv2.imread(str(key_path)))
    #graph_util.plot_mask(img, answer_key_mask, Path(str(key_path).replace(".jpg", "_mask_out.jpg")))

    #get the mask out of the test method.    

    test_mask = plant_finder.find_plants(cv2.imread(str(src_path)))

    graph_util.plot_blobs(img, Path(str(src_path).replace(".jpg", "_test_mask_out.jpg")), test_mask)

    #check accuracy

    accuracy = du.compare_two_masks(test_mask, answer_key_mask) 



'''def test_more():
    test_cases = [
        (
            "badmint.jpg",
            ((31, 50, 50), (75, 255, 200)),  # green bounds
            22,
        ),
        (
            "lettuce_8.jpg",
            ((31, 50, 50), (75, 255, 200)), 
            25,
        ),
    ]

    for filename, color_bounds, tolerance in test_cases:
        src = IMG_DIR / filename
        dst = IMG_DIR / filename.replace(".jpg", "_grown_out.png")
        run_pipeline_on_image(src, dst, color_bounds, color_tolerance=tolerance)'''



#if __name__ == "__main__":
#    test_blobber("tests/testimages/IMG_3387.jpg")
#    test_pipeline_on_image("tests/testimages/IMG_3387.jpg")

#    test_blobber("tests/testimages/IMG_3398.jpg")
#    test_pipeline_on_image("tests/testimages/IMG_3398.jpg")

#process_image_key("tests/testimages/plant_detect_accuracy_test/IMG_3397_key.jpg")
#test_accuracy("tests/testimages/plant_detect_accuracy_test/IMG_3397.jpg", "tests/testimages/plant_detect_accuracy_test/IMG_3397_key.jpg")