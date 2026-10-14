import cv2
import numpy as np
from collections import deque

def select_black_pixels(image):
    lower_val = np.array([0, 0, 0])
    upper_val = np.array([179, 255, 127])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_val, upper_val)

    return mask

def compare_two_masks(test_mask, answer_key_mask):
    accuracy = 0.0

    if test_mask.contourArea() == 0 or answer_key_mask.contourArea() == 0:
        print("At least one mask is empty.")
        accuracy = 0.0
    
    if test_mask.contourArea() == answer_key_mask.contourArea() and test_mask.shape == answer_key_mask.shape:
        print("Masks have the exact same area and shape.")
        accuracy = 100.0

    if test_mask.shape != answer_key_mask.shape or test_mask.contourArea() != answer_key_mask.contourArea():
        errors = cv2.bitwise_xor(test_mask, answer_key_mask)
        accuracy = (answer_key_mask.contourArea() - errors.contourArea()) / answer_key_mask.contourArea() * 100
        print (f"Accuracy: {accuracy:.2f}%")

    else:
        accuracy = 0.0

    return accuracy