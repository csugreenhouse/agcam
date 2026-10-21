import cv2
import numpy as np
from collections import deque
import importlib

plant_finder = importlib.import_module("plant_finders.legacyHSV_plant_finder")

def select_white_pixels(image):
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(image, lower, upper)

    return mask

def get_contours_from_mask(mask):
    #src: https://stackoverflow.com/questions/36721936/opencv-finding-contours-on-numpy-array
    _, t2 = cv2.threshold(mask,127,255,0)
    contours, hierarchy = cv2.findContours(t2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compare_two_masks(test_mask, key_mask):
    accuracy = 0.0

    answer_key_mask = cv2.bitwise_not(key_mask)

    if cv2.countNonZero(test_mask) == 0 ^ cv2.countNonZero(answer_key_mask) == 0:
        print("One mask is empty")
        accuracy = 0.0
    
    if cv2.countNonZero(test_mask) == cv2.countNonZero(answer_key_mask) and test_mask.shape == answer_key_mask.shape:
        print("Masks have the exact same area and shape.")
        accuracy = 100.0

    if test_mask.shape != answer_key_mask.shape or cv2.countNonZero(test_mask) != cv2.countNonZero(answer_key_mask):
        errors = cv2.bitwise_xor(test_mask, answer_key_mask)
        accuracy = (cv2.countNonZero(answer_key_mask) - cv2.countNonZero(errors)) / cv2.countNonZero(answer_key_mask) * 100
        print (f"Accuracy: {accuracy:.2f}%")

    else:
        accuracy = 0.0

    return accuracy