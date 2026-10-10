# LEGACY CODE WRITTEN BY TYLER CARVER, IMPORTED TO NEW PROJECT

import importlib
from pathlib import Path
import os
import pytest
from pytest import approx
import array
import cv2
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

height_request = importlib.import_module("processing.height_request")
scanner_util = importlib.import_module("utils.reference_tag_util")
graph_util = importlib.import_module("utils.graph_util")

def test_apritag_test01():
    src = "tests/testimages/TEST01.jpg"
    dst = "tests/testimages/TEST01_out.jpg"
    
    image = cv2.imread(str(src))
    tag_info = scanner_util.scan_reference_tags(image)[0]
    tag_info["data"] == 1
    assert "corners" in tag_info, "scan_reference_tags() returned no 'corners'"
    corners = tag_info["corners"]
    assert(type(corners)==dict)
    tl = corners["top_left"]
    tr = corners["top_right"]
    br = corners["bottom_right"]
    bl = corners["bottom_left"] 
    #run asserts for github side testing and print success after all run well
    assert tl == approx((267.1820373534919, 244.1751403808336),rel=1e-3,abs=0.5)
    assert tr == approx((393.2828063964598, 250.85649108888924),rel=1e-3,abs=0.5)
    assert br == approx((387.81927490236626, 374.93643188478995),rel=1e-3,abs=0.5)
    assert bl == approx((262.30166625978956, 369.2857971191188),rel=1e-3,abs=0.5)
    print(f"test_apritag_test01 passed successfully") 

    print(f"begin tag plotter")

    graph_util.plot_reference_tag(image,dst, tag_info)

'''def test_apritag_test02():
    src = IMG_DIR / "TEST02.jpg"
    dst = IMG_DIR / "TEST02_out.png"
    
    image = cv2.imread(str(src))
    tag_info = scanner_util.scan_reference_tags(image, test_camera_parameters, test_connection)[0]

    assert "corners" in tag_info, "scan_reference_tags() returned no 'corners'"
    corners = tag_info["corners"]
    tl = corners["top_left"]
    tr = corners["top_right"]
    br = corners["bottom_right"]
    bl = corners["bottom_left"] 
    
    assert tl == approx((558.5017700195309, 260.1452331542965),rel=1e-3,abs=0.5)   
    assert tr == approx((636.921752929687, 263.3018188476566),rel=1e-3,abs=0.5)
    assert br == approx((632.7066650390628, 341.7890319824224),rel=1e-3,abs=0.5)
    assert bl == approx((554.6207275390632, 338.9519042968744),rel=1e-3,abs=0.5)

def test_apritag_test03():
    src = IMG_DIR /"TEST03.jpg"
    dst = IMG_DIR / "TEST03_out.png"
    
    image = cv2.imread(str(src))
    tag_info = scanner_util.scan_reference_tags(image, test_camera_parameters, test_connection)[0]

    assert "corners" in tag_info, "scan_reference_tags() returned no 'corners'"
    corners = tag_info["corners"]
    tl = corners["top_left"]
    tr = corners["top_right"]
    br = corners["bottom_right"]
    bl = corners["bottom_left"] 
    
    assert tl == approx((442.08078002929653, 258.8964538574215),rel=1e-3,abs=0.5)
    assert tr == approx((497.32519531249955, 260.3209533691411),rel=1e-3,abs=0.5)
    assert br == approx((495.6865844726565, 315.3803405761721),rel=1e-3,abs=0.5)
    assert bl == approx((440.6880798339846, 313.9172973632811), rel=1e-3,abs=0.5)'''

def test_apritag_testNONE():
    src = "tests/testimages/TESTNONE.jpg"
    image = cv2.imread(str(src))
    
    with pytest.raises(ValueError) as excinfo:
        tag_info = scanner_util.scan_reference_tags(image)
    
    assert "No reference tags at all have been found" in str(excinfo.value)
    print(f"test_apritag_testNONE passed successfully")

if __name__ == "__main__":
    test_apritag_testNONE()
    test_apritag_test01()
    print(f"All apriltag detection tests passed successfully")