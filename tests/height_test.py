from pathlib import Path
import os
import pytest
import cv2
import warnings
import sys
import importlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

height_request = importlib.import_module("processing.height_request")
scanner_util = importlib.import_module("utils.reference_tag_util")
graph_util = importlib.import_module("utils.graph_util")
#database_util = importlib.import_module("database.database")

IMG_DIR = Path(__file__).parent.parent / "images"
GRPH_DIR = Path(__file__).parent / "graphs"

#test_connection = database_util.open_connection_to_test_database()
#test_camera_parameters = database_util.get_available_camera_parameters_from_database(test_connection)[0]

def test_1():
    run_test("tests/testimages/basil_4.jpg", .28) #looks like 28cm
 
def run_test(image_path, expected_height):
    image = cv2.imread(str(image_path))

    reference_tags = scanner_util.scan_reference_tags(image)
    height_response = height_request.height_request(image, reference_tags)
    #to get minimum running product, did not graph, am trusting message for now, add graphing later.
    #problem with old code is that it is not fully modular and these height tests are dependent on the database
    #graph_util.plot_height_request_response(image,str(Path(image_path).with_name(Path(image_path).stem + "_out.jpg")),height_response)
    
    if expected_height is not None:
        assert height_response[0]["estimated_height"] == pytest.approx(expected_height, abs=0.5), f"Estimated height for plant 1 in image {str(image_path)} is {height_response[0]['estimated_height']} , but expected {expected_height}"
    
    reference_tag = reference_tags[0]

    out_path = str(Path(image_path).with_name(Path(image_path).stem + "_heighttest_out.jpg"))

    graph_util.plot_height(image, out_path, reference_tag)

if __name__ == "__main__":
    test_1()
    print(f"All height tests passed successfully")