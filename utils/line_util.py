import cv2
import numpy as np

def get_intercept_of_lines(equation1, equation2):
    slope1, intercept1 = equation1
    slope2, intercept2 = equation2
    
    if slope1 == slope2:
        raise ValueError("Lines are parallel - no intersection point")
    
    x_intercept = (intercept2 - intercept1) / (slope1 - slope2)
    y_intercept = slope1 * x_intercept + intercept1
    
    return x_intercept, y_intercept

def get_vertical_line(x_coordinate):
    slope = float('inf')
    intercept = x_coordinate
    return slope, intercept

def get_horizontal_line(y_coordinate):
    slope = 0
    intercept = y_coordinate
    return slope, intercept

def get_equation_of_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 == 0:
        raise ValueError("Vertical line - slope is undefined")
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def fractional_height_between_lines(equation_top, equation_bottom, coordinate):
    slope_top, intercept_top = equation_top
    slope_bottom, intercept_bottom = equation_bottom
    x, y = coordinate
    
    y_top = slope_top * x + intercept_top
    y_bottom = slope_bottom * x + intercept_bottom
    
    if y_bottom - y_top == 0:
        raise ValueError("Top and bottom lines are the same - cannot compute fractional height")
    
    fractional_height = 1 - (y - y_top) / (y_bottom - y_top)
    return fractional_height  

def fractional_width_between_lines(equation_left, equation_right, coordinate):
    slope_l, intercept_l = equation_left
    slope_r, intercept_r = equation_right
    x, y = coordinate
    
    if slope_l == float('inf'):
        x_left = intercept_l
    else:
        x_left = (y - intercept_l) / slope_l
        
    if slope_r == float('inf'):
        x_right = intercept_r
    else:
        x_right = (y - intercept_r) / slope_r
    
    if x_right - x_left == 0:
        raise ValueError("Left and right lines are the same - cannot compute fractional width")
    
    fractional_width = (x - x_left) / (x_right - x_left)
    return fractional_width