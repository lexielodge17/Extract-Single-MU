#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:33:04 2026

@author: lexielodge
"""

import cv2
import json

def find_roi(VIDEO_PATH):
    
    """
    find the region of interest for the video
    """
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Could not open video"
    
    ret, frame = cap.read()
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(frame.shape)

    r = cv2.selectROI("Select ROI", frame, fromCenter=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cap.release()
    
    return r

def calculate_pixel_size(width, depth, w, h):
    
    """
    calculate the pixel dimensions based on the width and depth of US video file
    vs the known width and depth of US probe 
    """
    
    pixel_size_x = width/w
    pixel_size_y = depth/h
    
    return pixel_size_x, pixel_size_y

# ===================== USER PARAMETERS =====================

VIDEO_ID = "20260428_AW_RTA_SMU"
VIDEO_PATH = "{}.avi". format(VIDEO_ID)
width = 30
depth = 25

# ===================== Find ROI Full US frame =====================

ROI_US = find_roi(VIDEO_PATH)
x0_US, y0_US, w_US, h_US = ROI_US 


roi_US = {
    "x0": x0_US,
    "y0": y0_US,
    "w": w_US,
    "h": h_US
}

# ===================== Calculate Pixel Dimensions =====================

PIXEL_SIZE_X_MM, PIXEL_SIZE_Y_MM = calculate_pixel_size(width, depth, w_US, h_US)
PIXEL_AREA_MM2 = PIXEL_SIZE_X_MM * PIXEL_SIZE_Y_MM

pixels = {
    "pixel x (mm)": PIXEL_SIZE_X_MM,
    "pixel y (mm)": PIXEL_SIZE_Y_MM,
    "pixel area (mm2)": PIXEL_AREA_MM2
}

# ===================== Find ROI Full US frame =====================

# Select area around where motor unit most likely to be

ROI_MOTOR_UNIT = find_roi(VIDEO_PATH)
x0_MU, y0_MU, w_MU, h_MU = ROI_MOTOR_UNIT


roi_MU = {
    "x0": x0_MU,
    "y0": y0_MU,
    "w": w_MU,
    "h": h_MU
}

# ===================== Combine data into dict and save as txt file =====================

roi_pixels = {
    "ROI US": roi_US,
    "ROI MU": roi_MU,
    "Pixels": pixels
}
    
with open('roi_pixels {}.json'.format(VIDEO_ID), 'w') as f:
    json.dump(roi_pixels, f, indent=4)


