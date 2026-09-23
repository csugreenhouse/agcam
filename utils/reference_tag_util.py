import cv2
import numpy as np
import warnings
import sys
sys.path.append('/mnt/db/agcam')
import apriltag

def scan_raw_tags(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    options = apriltag.DetectorOptions(families="tag25h9")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    return results


def scan_reference_tags(image, camera_parameters, conn, current_only=False):
    results = scan_raw_tags(image)
    
    DECISION_MARGIN = 40.0

    reference_tags = []

    for raw_tag in results:
        #print(f"TAG ID {tag.tag_id} with decision margin {tag.decision_margin}")
        if (raw_tag.decision_margin>DECISION_MARGIN):
            reference_tag = make_reference_tag(raw_tag,camera_parameters,conn, current_only)
            reference_tags.append(reference_tag)    
    
    if (len(reference_tags)==0):
        if (len(results)!=0):
            raise ValueError(f"No Valid reference tag has been detected, but a non valid one has been found")
        else:
            raise ValueError(f"No reference tags at all have been found")
  
    return reference_tags

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
