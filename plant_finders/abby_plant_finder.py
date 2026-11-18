"""
Copyright         : 2026 Tyler Carver & Kai Brennan
File Name         : legacyHSV_plant_finder.py
Description       : legacy plant finder from Senior Design Spring 2026. Color based detection, primarily.
                  : pipeline: 0: bring in original image 1: find green blobs, 2: remove blobs with odd aspect ratio & discard blobs far from the largest anchor or too small, 3: grow regions from seed mask (final output)
 
Revision History  :
Date		Author 			Comments
------------------------------------------------------------------
LEGACY FILE, OLD HISTORY UNTRACKED
14 Oct 2026 Kai Brennan Refactored method according to styleguide and commented.
"""

# IMPORTS
import cv2
import numpy as np
from collections import deque

# VARIABLES (Control Panel)
GREEN_LOWER  = np.array([ 31, 50, 50], dtype="uint8")
GREEN_UPPER  = np.array([ 75, 255, 220], dtype="uint8")
GREEN_BOUNDS = (GREEN_LOWER, GREEN_UPPER)

def find_plants(image_file_path, color_bounds=GREEN_BOUNDS):
    """
    fully assembled pipeline method for plant detection
    0: bring in original image 1: find green blobs, 2: remove blobs with odd aspect ratio & discard blobs far from the largest anchor or too small, 3: grow regions from seed mask (final output)

    Args:
        image: the image to detect plants in

    Returns:
        grown_mask: OpenCV mask of detected plants
    """

    # Error Checking for missing files 

    image = cv2.imread(str(image_file_path))
    if image is None:
        raise FileNotFoundError(f"Error: Could not find file{image_file_path}")
    

    def compute_plant_mask(image):
        H,W = image.shape[:2]

        smooth = cv2.bilateraFilter(image, d=7, sigmaColor=35, sigmaSpace=35)
        img_f = smooth.astype(np.float32)
        B,G,R = cv2.split(img_f)
        #excess of color (green and red)
        excess_gr = 2.0 * G - R - B
        excess_gr_red = 3.0 * G - 2.4 * R - B
        #shadow control
        norm_diff = (G -R)/(G+R+1.0)
        glob_diff = (G - B) / (G + B + 1.0)
        #greenhouse lighting
        lab =cv2.cvtColor(smooth, cv2.COLOR_BGRLAB). astype(np.float32)
        L_chan, a_chan, b_chan = cv2.split(lab)
        green_a = 128.0 - a_chan
        yellow_b = b_chan - 128.0

        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
        H_chan, S_chan, V_chan = cv2.split(hsv)

        #create a likelyhood score
        s_a = np.clip((green_a - 2.5) / 18.0, 0.0, 1.0)
        s_norm_diff = np.clip((norm_diff + 0.01)/ 0.18, 0.0, 1.0)
        s_excess_gr = np.clip((excess_gr - 3.0)/ 32.0, 0.0,1.0)
        s_excess_gr_red = np.clip((excess_gr_red + 5)/ 40.0, 0.0, 1.0)

        plant_score = 0.40 * s_a + 0.25 * s_norm_diff + 0.20 * s_excess_gr + 0.15 * s_excess_gr_red

        light_leaf = (
            (H_chan >= 30) & (H_chan <= 82) & (S_chan >= 40) & (V_chan >= 35) & (excess_gr > 10.0) & (green_a > 6.5)
        )
        dark_leaf = ( 
            (H_chan>=32)& (H_chan <=85) & (S_chan >= 35) & (V_chan>= 20)& (norm_diff > 0.66) 
        )
        raw_plant_mask = np.where(light_leaf | dark_leaf, 255, 0). astype(np.uint8)

        outside_mask = (
            (green_a < -1.5)| ((V_chan < 22)  &  (green_a < 3.5) (V_chan > 215) & (S_chan < 35))
        )
        return plant_score.astype(np.float32), raw_plant_mask, outside_mask
    

    def filter_blobs(plant_mask, min_area=80, min_aspect_ratio=0.12, max_aspect_ratio=7.5):
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        blobs = []

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label,cv2.CC_STAT_HEIGHT]
            aspect = float(w)/float(h) if h > 0 else 0.0
            