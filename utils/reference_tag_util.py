import cv2
import numpy as np
import warnings
import sys
sys.path.append('/mnt/db/agcam')
import pupil_apriltags as apriltag
from pupil_apriltags import Detector
import importlib

import database.database as database_util


def scan_raw_tags(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    detector = Detector(families="tag25h9")
    results = detector.detect(gray)
    return results


def scan_reference_tags(image, current_only=False):
    results = scan_raw_tags(image)
    
    DECISION_MARGIN = 40.0

    reference_tags = []

    for raw_tag in results:
        #print(f"TAG ID {tag.tag_id} with decision margin {tag.decision_margin}")
        if (raw_tag.decision_margin>DECISION_MARGIN):
            reference_tag = make_reference_tag(raw_tag, current_only)
            reference_tags.append(reference_tag)    
    
    if (len(reference_tags)==0):
        if (len(results)!=0):
            raise ValueError(f"No Valid reference tag has been detected, but a non valid one has been found")
        else:
            raise ValueError(f"No reference tags at all have been found")
  
    return reference_tags

def make_reference_tag(raw_april_tag, current_only, scale=None, views=None):
    # if no connection is passed, ensure that scales and views are populated.
    '''if (conn is None):
        if scale is None or views is None:
             raise ValueError("both view and scale need to be provided if you dont pass a connection")
    # pupil_apriltags detection fields:
    #   raw_april_tag.tag_id, raw_april_tag.corners, raw_april_tag.center
    tag_id = int(raw_april_tag.tag_id)
    
    scale = scale #if scale is not None else database_util.get_tag_scale_from_database(conn, tag_id)
    views = views #if views is not None else database_util.get_tag_views_from_database(conn, tag_id, current_only=current_only)
    '''
    tag_id = int(raw_april_tag.tag_id)
    # Ensure numpy arrays -> tuples (JSON safe)
    corners_np = np.asarray(raw_april_tag.corners, dtype=np.float32)
    center_np = np.asarray(raw_april_tag.center, dtype=np.float32)
    
    corners_sorted = sort_corners(corners_np)
    
    corners = {
        "top_left": tuple(corners_sorted[0]),
        "top_right": tuple(corners_sorted[1]),
        "bottom_right": tuple(corners_sorted[2]),
        "bottom_left": tuple(corners_sorted[3]),
    }
    
    center = tuple(center_np)

    # displacement_d, displacement_z, displacement_x, displacement_y = calculate_displacement(
    #     camera_parameters, scale, center, corners
    # )
    '''if not views:
        raise ValueError("Views is None or empty")

    for view in views:
        if view["image_bound_x_low"] > view["image_bound_x_high"]:
            raise ValueError("image bounds x in a view are switched")
        if view["image_bound_y_low"] > view["image_bound_y_high"]:
            raise ValueError("image bounds y in a view are switched")'''

    tag = {
        "data": tag_id,
        "tag_type": "referencetag",
        "center": center,
        "corners": corners,
        #"scale_units_m": float(scale),
        #"views": views,
    }
    return tag

'''def filter_reference_tags_by_view_type(reference_tags, view_type):
    filtered_tags = []
    for tag in reference_tags:
        filtered_views = [view for view in tag["views"] if view["view_type"] == view_type]
        if filtered_views:
            new_tag = tag.copy()
            new_tag["views"] = filtered_views
            filtered_tags.append(new_tag)
    return filtered_tags'''

def sort_corners(corners):
    """
    Sort corners in the order: top_left, top_right, bottom_right, bottom_left
    """
    corners = np.array(corners)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    top_left = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    top_right = corners[np.argmin(diff)]
    bottom_left = corners[np.argmax(diff)]

    return [top_left, top_right, bottom_right, bottom_left]