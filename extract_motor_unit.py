#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:03:19 2026

@author: lexielodge
"""

import cv2
import numpy as np
from collections import deque
import csv
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import pdist
import json
import sys

def top_hat_filter(frame, radius=15):
    """
    White top-hat filter for ultrasound images. Spatial filter.
    radius: should be larger than the expected twitch feature size.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius, radius)
    )

    filtered = cv2.morphologyEx(
        frame,
        cv2.MORPH_TOPHAT,
        kernel
    )

    return filtered

def temporal_bandpass(frame,
                      lowcut,
                      highcut,
                      fs,
                      order):
    
    """Temporal bandpass filter"""
    
    nyquist = fs / 2

    b, a = butter(
        order,
        [lowcut / nyquist, highcut / nyquist],
        btype='band'
    )

    return filtfilt(b, a, frame, axis=0)

def estimate_baseline_noise(cap, x0, y0, x1, y1, n_frames=50):
    """Sample early frames (assumed quiescent) to estimate per-video noise floor."""
    variances = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        variances.append(roi)
    stack = np.stack(variances, axis=0)
    noise_var_map = np.var(stack, axis=0)
    return np.median(noise_var_map), np.std(noise_var_map)

def get_expected_search_window(stim_time_sec, fps,
                                 delay_min_sec=0.01, delay_max_sec=0.15):
    """Restrict twitch search to physiologically plausible window post-stimulus."""
    start_frame = int((stim_time_sec + delay_min_sec) * fps)
    end_frame = int((stim_time_sec + delay_max_sec) * fps)
    return start_frame, end_frame

def find_values(obj, target_key, results=None):
    """
    Find values from a json file
    """
    if results is None:
        results = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                results.append(value)
            find_values(value, target_key, results)
    elif isinstance(obj, list):
        for item in obj:
            find_values(item, target_key, results)
    
    return results

def extract_MU_contours(cap, x0, x1, y0, y1, FPS, WINDOW_SIZE, 
                        CONFIRM_WINDOW, AREA_RATIO_THRESH, noise_median, noise_std):
    
    roi_area = (x1 - x0) * (y1 - y0)
    
    # ---- Twitch area tracking ----

    peak_variance_energy = 0
    peak_var_map = None

    twitch_areas = []   # Stores (start_time, end_time, area_mm2, feret diameters)
    all_contours_masks = []  # list of binary masks, one per confirmed twitch
    
    buffer = deque(maxlen=WINDOW_SIZE)
    detection_history = deque(maxlen=CONFIRM_WINDOW)

    frame_idx = 0
    twitch_events = []
    in_twitch = False
    current_contour = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Convert to grayscale if needed
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Preprocessing
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        buffer.append(roi)

        twitch_detected = False

        if len(buffer) == WINDOW_SIZE:
            
            stack = np.stack(buffer, axis=0)
                
            # Assert whether bandpass/top-hat needed
            if noise_median > 50 or noise_std > 100: #noise_median = 25 prev
                # Temporal bandpass: axis=0 is time, filtered independently per pixel
                stack_bp = temporal_bandpass(stack, lowcut=4, highcut=25, fs=FPS, order=2)
        
                # Temporal variance with Gaussian blurring
                var_map = np.var(stack_bp, axis=0)
                var_map = cv2.GaussianBlur(var_map, (11,11), 2.0) 
                
                # Apply top-hat on the final 2D map
                var_map = top_hat_filter(var_map.astype(np.float32), radius=15)
                
            else:
                # Temporal variance with Gaussian blurring
                var_map = np.var(stack, axis=0)
                var_map = cv2.GaussianBlur(var_map, (11,11), 2.0) 
            
            # Compute total variance energy
            total_variance_energy = np.sum(var_map)

            # If currently inside twitch, track peak frame
            if in_twitch:
                if total_variance_energy > peak_variance_energy:
                    peak_variance_energy = total_variance_energy
                    peak_var_map = var_map.copy()
            
            # Create heatmap
            heat = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX)
            heat = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
            
            # ---- Overlay detection markers ----
            
            current_time = frame_idx / FPS
            
            # Timestamp
            cv2.putText(
                heat,
                f"{current_time:.3f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            if in_twitch:
                # Text label
                cv2.putText(
                    heat,
                    "TWITCH ACTIVE",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            
                # Green border
                cv2.rectangle(
                    heat,
                    (0, 0),
                    (w-1, h-1),
                    (0, 255, 0),
                    3
                )
            
            # Draw motor unit contour if available
            if current_contour is not None:
                cv2.drawContours(
                    heat,
                    [current_contour],
                    -1,
                    (255, 255, 255),   # white contour
                    2
                )
            
            cv2.imshow("Variance Heatmap", heat)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # ------------------------------------
            
            # Mean + MAD
            baseline = np.median(var_map)
            mad = np.median(np.abs(var_map - baseline))
            k = 6 if noise_std / noise_median > 3 else 4
            thresh = baseline + k * mad 
            
            motion_mask = var_map > thresh

            motion_area = np.sum(motion_mask)
            twitch_detected = (motion_area > AREA_RATIO_THRESH * roi_area)

        detection_history.append(twitch_detected)

        # Temporal confirmation
        if sum(detection_history) >= MIN_DETECTIONS and not in_twitch:
            in_twitch = True
            twitch_start_time = frame_idx / FPS
            twitch_events.append(("START", twitch_start_time))
            
            # Reset peak tracking
            peak_variance_energy = 0
            peak_var_map = None

        if sum(detection_history) < MIN_DETECTIONS and in_twitch:
            in_twitch = False
            twitch_end_time = frame_idx / FPS
            twitch_events.append(("END", twitch_end_time))
            
            # ---- Compute twitch area at peak frame ----
            if peak_var_map is not None:
                
                # Robust threshold (Median + MAD)
                baseline = np.median(peak_var_map)
                mad = np.median(np.abs(peak_var_map - baseline))
                k = 6 if noise_std / noise_median > 3 else 4
                thresh = baseline + k * mad 
        
                motion_mask = peak_var_map > thresh
                
                # Morphological cleanup
                kernel = np.ones((3, 3), np.uint8)
                motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8),
                                               cv2.MORPH_OPEN, kernel)
                motion_mask = cv2.morphologyEx(motion_mask,
                                               cv2.MORPH_CLOSE, kernel)
        
                # Connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    motion_mask, connectivity=8)
        
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    largest_mask = (labels == largest_label).astype(np.uint8)
                    
                    twitch_area_pixels = stats[largest_label, cv2.CC_STAT_AREA]
                        
                    if twitch_area_pixels >= MIN_AREA_PIXELS:
                        contours, _ = cv2.findContours(
                            largest_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                    
                        if contours:
                            current_contour = cv2.convexHull(contours[0])
                            
                            # Rasterize this twitch's contour into a full-ROI binary mask
                            twitch_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.drawContours(twitch_mask, [current_contour], -1, 1, thickness=cv2.FILLED)
                            all_contours_masks.append(twitch_mask)

                            rect = cv2.minAreaRect(current_contour)
                            width_px, height_px = rect[1]
                            
                            #Rescale into physical space
                            scaled_contour = current_contour.astype(np.float32)

                            scaled_contour[:, :, 0] *= PIXEL_SIZE_X_MM
                            scaled_contour[:, :, 1] *= PIXEL_SIZE_Y_MM
                            
                            rect2 = cv2.minAreaRect(scaled_contour)
                            width_mm, height_mm = rect2[1]
                            
                            #Compute Feret diameters
                            hull_pts = scaled_contour.reshape(-1, 2)
                            max_feret_mm = pdist(hull_pts).max()
                            min_feret_mm = min(width_mm, height_mm)

                        else:
                            current_contour = None
                
                else:
                    twitch_area_pixels = 0
                    current_contour = None
                
                
                # Convert to mm²
                area_mm2 = twitch_area_pixels * PIXEL_AREA_MM2
            
                twitch_areas.append(
                    (twitch_start_time, twitch_end_time, area_mm2, max_feret_mm, min_feret_mm)
                )
        
        
            # Reset peak trackers
            peak_variance_energy = 0
            peak_var_map = None
            
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return twitch_areas, all_contours_masks
    
def extract_MU_contours_locked(cap, x0, x1, y0, y1, FPS, WINDOW_SIZE, 
                               CONFIRM_WINDOW, AREA_RATIO_THRESH, noise_median, noise_std, stim_times_sec):
    
    roi_area = (x1 - x0) * (y1 - y0)
    
    # ---- Twitch area tracking ----

    peak_variance_energy = 0
    peak_var_map = None

    twitch_areas = []   # Stores (start_time, end_time, area_mm2, feret diameters)
    all_contours_masks = []  # list of binary masks, one per confirmed twitch
    
    for stim_t in stim_times_sec:
        start_frame, end_frame = get_expected_search_window(
            stim_t, FPS, delay_min_sec=0.01, delay_max_sec=0.5
        )
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        buffer = deque(maxlen=WINDOW_SIZE)
        detection_history = deque(maxlen=CONFIRM_WINDOW)
        
        frame_idx = 0
        twitch_events = []
        in_twitch = False
        current_contour = None
        
        for i in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Pre-processing
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            roi = frame[y0:y1, x0:x1].astype(np.float32)
            buffer.append(roi)
            
            twitch_detected = False

            if len(buffer) == WINDOW_SIZE:
                stack = np.stack(buffer, axis=0)
                    
                # Assert whether bandpass/top-hat needed
                if noise_median > 50 or noise_std > 100: #noise_median = 25 prev
                    # Temporal bandpass: axis=0 is time, filtered independently per pixel
                    stack_bp = temporal_bandpass(stack, lowcut=4, highcut=25, fs=FPS, order=2)
            
                    # Temporal variance with Gaussian blurring
                    var_map = np.var(stack_bp, axis=0)
                    var_map = cv2.GaussianBlur(var_map, (11,11), 2.0) 
                    
                    # Apply top-hat on the final 2D map
                    var_map = top_hat_filter(var_map.astype(np.float32), radius=15)
                    
                else:
                    # Temporal variance with Gaussian blurring
                    var_map = np.var(stack, axis=0)
                    var_map = cv2.GaussianBlur(var_map, (11,11), 2.0) 
                    
                # Compute total variance energy
                total_variance_energy = np.sum(var_map)
                

                # If currently inside twitch, track peak frame
                if in_twitch and total_variance_energy > peak_variance_energy:
                    peak_variance_energy = total_variance_energy
                    peak_var_map = var_map.copy()

                # Create heatmap
                heat = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX)
                heat = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
                
                # ---- Overlay detection markers ----
                    
                current_time = (start_frame + frame_idx) / FPS
                
                # Timestamp
                cv2.putText(
                    heat,
                    f"{current_time:.3f}s",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                if in_twitch:
                    # Text label
                    cv2.putText(
                        heat,
                        "TWITCH ACTIVE",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )
                
                    # Green border
                    cv2.rectangle(
                        heat,
                        (0, 0),
                        (w-1, h-1),
                        (0, 255, 0),
                        3
                    )
                
                # Draw motor unit contour if available
                if current_contour is not None:
                    cv2.drawContours(
                        heat,
                        [current_contour],
                        -1,
                        (255, 255, 255),   # white contour
                        2
                    )
                
                cv2.imshow("Variance Heatmap", heat)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # ------------------------------------
                
                # Mean + MAD
                baseline = np.median(var_map)
                mad = np.median(np.abs(var_map - baseline))
                k = 6 if noise_std / noise_median > 3 else 4
                thresh = baseline + k * mad 
                motion_mask = var_map > thresh
                motion_area = np.sum(motion_mask)
                
                twitch_detected = motion_area > AREA_RATIO_THRESH * roi_area
                
            detection_history.append(twitch_detected)

            if sum(detection_history) >= MIN_DETECTIONS and not in_twitch:
                in_twitch = True
                peak_variance_energy = 0
                twitch_start_time = (start_frame + frame_idx) / FPS
                twitch_events.append(("START", twitch_start_time))
                peak_var_map = None
                
                in_twitch = True
                
            
            if sum(detection_history) < MIN_DETECTIONS and in_twitch:
                in_twitch = False
                twitch_end_time = (start_frame + frame_idx)/ FPS
                twitch_events.append(("END", twitch_end_time))
                
                # ---- Compute twitch area at peak frame ----
                if peak_var_map is not None:
                    
                    # Robust threshold (Median + MAD)
                    baseline = np.median(peak_var_map)
                    mad = np.median(np.abs(peak_var_map - baseline))
                    k = 6 if noise_std / noise_median > 3 else 4
                    thresh = baseline + k * mad 
            
                    motion_mask = peak_var_map > thresh
                    
                    # Morphological cleanup
                    kernel = np.ones((3, 3), np.uint8)
                    motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8),
                                                   cv2.MORPH_OPEN, kernel)
                    motion_mask = cv2.morphologyEx(motion_mask,
                                                   cv2.MORPH_CLOSE, kernel)
            
                    # Connected components
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                        motion_mask, connectivity=8)
            
                    if num_labels > 1:
                        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        largest_mask = (labels == largest_label).astype(np.uint8)
                        
                        twitch_area_pixels = stats[largest_label, cv2.CC_STAT_AREA]
                            
                        if twitch_area_pixels >= MIN_AREA_PIXELS:
                            contours, _ = cv2.findContours(
                                largest_mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                        
                            if contours:
                                current_contour = cv2.convexHull(contours[0])
                                
                                # Rasterize this twitch's contour into a full-ROI binary mask
                                twitch_mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.drawContours(twitch_mask, [current_contour], -1, 1, thickness=cv2.FILLED)
                                all_contours_masks.append(twitch_mask)

                                rect = cv2.minAreaRect(current_contour)
                                width_px, height_px = rect[1]
                                
                                #Rescale into physical space
                                scaled_contour = current_contour.astype(np.float32)

                                scaled_contour[:, :, 0] *= PIXEL_SIZE_X_MM
                                scaled_contour[:, :, 1] *= PIXEL_SIZE_Y_MM
                                
                                rect2 = cv2.minAreaRect(scaled_contour)
                                width_mm, height_mm = rect2[1]
                                
                                #Compute Feret diameters
                                hull_pts = scaled_contour.reshape(-1, 2)
                                max_feret_mm = pdist(hull_pts).max()
                                min_feret_mm = min(width_mm, height_mm)

                            else:
                                current_contour = None
                    
                    else:
                        twitch_area_pixels = 0
                        current_contour = None
                    
                    # Convert to mm²
                    area_mm2 = twitch_area_pixels * PIXEL_AREA_MM2
                
                    twitch_areas.append(
                        (twitch_start_time, twitch_end_time, area_mm2, max_feret_mm, min_feret_mm)
                    )
            
            
                # Reset peak trackers
                peak_variance_energy = 0
                peak_var_map = None
                
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)            
    
    return twitch_areas, all_contours_masks

def compute_average_mask(masks_list, agreement_thresh=0.5):
    """
    agreement_thresh: fraction of twitches that must include a pixel
    for it to count as part of the averaged motor unit region.
    0.5 = majority vote. Lower = more inclusive/union-like.
    Higher = stricter, closer to intersection.
    """
    stack = np.stack(masks_list, axis=0).astype(np.float32)
    prob_mask = np.mean(stack, axis=0)   # 0-1, fraction of twitches covering each pixel
    avg_mask = (prob_mask >= agreement_thresh).astype(np.uint8)
    return avg_mask, prob_mask

def compute_average_mask_adaptive(masks_list, start_thresh=0.5, min_thresh=0.1, step=0.1):
    stack = np.stack(masks_list, axis=0).astype(np.float32)
    prob_mask = np.mean(stack, axis=0)
    """
    computes average mask using an adaptive threshold (fraction of twitches that must include a pixel
    for it to count as part of the averaged motor unit region). Falls to a lower threshold when the 
    start_thresh value yields nothing. Ensures contour is always produced.
    The lower the returned thresh value, the less spatially consistent detected twitches were.
    """
    thresh = start_thresh
    while thresh >= min_thresh:
        avg_mask = (prob_mask >= thresh).astype(np.uint8)
        if np.count_nonzero(avg_mask) >= MIN_AREA_PIXELS:
            return avg_mask, prob_mask, thresh
        thresh -= step

    # Fall back to union if nothing else works
    avg_mask = (prob_mask > 0).astype(np.uint8)
    return avg_mask, prob_mask, 0.0

def calculate_mean_ferets_sd(twitch_areas):
    
    """
    Calculate the mean maximum and minimum feret diameters with standard deviation from 
    the calculated twitch areas
    """
    
    max_ferets = np.array([t[3] for t in twitch_areas])
    min_ferets = np.array([t[4] for t in twitch_areas])
    
    n = len(max_ferets)
    max_feret_mean, max_feret_sd = np.mean(max_ferets), np.std(max_ferets, ddof=1)
    min_feret_mean, min_feret_sd = np.mean(min_ferets), np.std(min_ferets, ddof=1)
    
    max_feret_sem = max_feret_sd / np.sqrt(n)
    min_feret_sem = min_feret_sd / np.sqrt(n)
    
    return n, max_feret_mean, max_feret_sd, max_feret_sem, min_feret_mean, min_feret_sd, min_feret_sem
    
    
def feret_from_mask(mask, px_x_mm, px_y_mm):
    """
    Calculate the feret diameter from the video mask
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan, np.nan
    contour = cv2.convexHull(max(contours, key=cv2.contourArea))
    scaled = contour.astype(np.float32)
    scaled[:, :, 0] *= px_x_mm
    scaled[:, :, 1] *= px_y_mm
    pts = scaled.reshape(-1, 2)
    if len(pts) < 2:
        return np.nan, np.nan
    d = pdist(pts)
    max_f = d.max()
    rect = cv2.minAreaRect(contour)
    min_f = min(rect[1])
    return max_f, min_f

def bootstrap_average_feret(masks_list, agreement_thresh, px_x_mm, px_y_mm,
                             n_bootstrap=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(masks_list)
    max_ferets, min_ferets = [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)          # sample with replacement
        resample = [masks_list[i] for i in idx]
        avg_mask, _ = compute_average_mask(resample, agreement_thresh)
        mf, mnf = feret_from_mask(avg_mask, px_x_mm, px_y_mm)
        if not np.isnan(mf):
            max_ferets.append(mf)
            min_ferets.append(mnf)

    return np.array(max_ferets), np.array(min_ferets)

# ===================== USER PARAMETERS =====================
VIDEO_ID = "20251021_020_L_TA_SMU"
VIDEO_PATH = "{}.avi". format(VIDEO_ID)

FPS = 108.856232019283   # sampling_rate in motion_metrics.json
WINDOW_SIZE = 16   # number of frames for sliding variance window
CONFIRM_WINDOW = 7    # 1 second
MIN_DETECTIONS = 5    # frames with motion in confirm window
AREA_RATIO_THRESH = 0.15   # % of ROI area required
LOCK_TO_STIM_TIMES = True # Set to True of False depending on whether you want to lock to stimulus times or not

# Load ROI and pixel data from json generated in calculate_roi_pixels   
try:
    with open('roi_pixels {}.json'.format(VIDEO_ID), "r", encoding="utf-8") as f:
        roi_data = json.load(f)
except FileNotFoundError:
    print("Error: roi and pixel data file not found. Please run calculate_roi_pixels.py for this video.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: invalid JSON — {e}")
    sys.exit(1)
  
    
print("Motor unit ROI and pixel dimensions loaded")

PIXEL_SIZE_X_MM = roi_data["Pixels"]["pixel x (mm)"]
PIXEL_SIZE_Y_MM = roi_data["Pixels"]["pixel y (mm)"]
MIN_AREA_PIXELS = 40 #the minimum number of connected pixels that must be active before a region is considered a real motor unit twitch.

PIXEL_AREA_MM2 = roi_data ["Pixels"]["pixel area (mm2)"]                                  

# ROI static, detected by choosing ROI
x0, y0, w, h = roi_data["ROI MU"]["x0"], roi_data["ROI MU"]["y0"], roi_data["ROI MU"]["w"], roi_data["ROI MU"]["h"]
x1, y1 = x0 + w, y0 + h

# ========================= MAIN DETECTION CODE ==================================

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Could not open video"

# Estimate baseline noise
noise_median, noise_std = estimate_baseline_noise(cap, x0, y0, x1, y1)

if LOCK_TO_STIM_TIMES:
   #Find stimulus times from motion_chunks
   with open('motion_chunks.json', 'r') as f:
       motion_chunks = json.load(f)

   stim_times_sec = find_values(motion_chunks, 'stimulus_time')
   
   twitch_areas, all_contours_masks = extract_MU_contours_locked(cap, x0, x1, 
                                                                 y0, y1, FPS, WINDOW_SIZE, 
                                                                 CONFIRM_WINDOW, AREA_RATIO_THRESH, 
                                                                 noise_median, noise_std, stim_times_sec,
                                                                 delay_min_sec=0.01, delay_max_sec=0.5)
   
else:
    twitch_areas, all_contours_masks = extract_MU_contours(cap, x0, x1, 
                                                           y0, y1, FPS, WINDOW_SIZE, 
                                                           CONFIRM_WINDOW, AREA_RATIO_THRESH, 
                                                           noise_median, noise_std)
        
# ========================= EXPORT AND POST-PROCESSING ==================================

# ---- Export twitch areas to csv ----   

with open("twitch_areas {}.csv".format(VIDEO_ID), "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Twitch Start (s)", "Twitch End (s)", "Area (mm2)", "Max Feret Diameter (mm)", "Min Feret Diameter (mm)"])
    for start, end, area, max_feret, min_feret in twitch_areas:
        writer.writerow([start, end, area, max_feret, min_feret])

print("Twitch areas exported to twitch_areas {}.csv".format(VIDEO_ID))
   
# ---- Average contours ----   

print(f"Number of twitch masks: {len(all_contours_masks)}")

#avg_mask, prob_mask = compute_average_mask(all_contours_masks, agreement_thresh=0.5)
try:
    avg_mask, prob_mask, thresh = compute_average_mask_adaptive(all_contours_masks, start_thresh=0.5, min_thresh=0.1, step=0.1)
except ValueError:
    print("Unable to calculate average mask")
    sys.exit(1)
    
# Extract the outer contour of the averaged mask
avg_contours, _ = cv2.findContours(avg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if avg_contours:
    average_contour = cv2.convexHull(max(avg_contours, key=cv2.contourArea))

    # --- Project onto first frame ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if first_frame.ndim == 3:
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_frame_bgr = first_frame.copy()
    else:
        first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
    
    heat = cv2.normalize(prob_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    
    blended_roi = cv2.addWeighted(first_frame_bgr[y0:y1, x0:x1], 0.6, heat_color, 0.4, 0)
    overlay_frame = first_frame_bgr.copy()
    overlay_frame[y0:y1, x0:x1] = blended_roi
    
    shifted_contour = average_contour + np.array([x0, y0])
    cv2.drawContours(overlay_frame, [shifted_contour], -1, (255, 255, 255), 2)
    
    cv2.imwrite("average_motor_unit_heatmap {}.png".format(VIDEO_ID), overlay_frame)
    print("Average motor unit heatmap created and saved to average_motor_unit_heatmap {}.png".format(VIDEO_ID))

# ---- Calculate mean feret with error ---------------------

n, max_feret_mean, max_feret_sd, max_feret_sem, min_feret_mean, min_feret_sd, min_feret_sem = calculate_mean_ferets_sd(twitch_areas)

print(f"Mean Max Feret: {max_feret_mean:.3f} ± {max_feret_sd:.3f} mm (SD, n={n}), SEM = {max_feret_sem:.3f}")
print(f"Mean Min Feret: {min_feret_mean:.3f} ± {min_feret_sd:.3f} mm (SD, n={n}), SEM = {min_feret_sem:.3f}")

boot_max, boot_min = bootstrap_average_feret(
    all_contours_masks, agreement_thresh=thresh,
    px_x_mm=PIXEL_SIZE_X_MM, px_y_mm=PIXEL_SIZE_Y_MM,
    n_bootstrap=1000
)

max_feret_point = feret_from_mask(avg_mask, PIXEL_SIZE_X_MM, PIXEL_SIZE_Y_MM)[0]
ci_low, ci_high = np.percentile(boot_max, [2.5, 97.5])

print(f"Average contour max Feret: {max_feret_point:.3f} mm, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")

with open("average_ferets {}.txt".format(VIDEO_ID), "w") as txtfile:
    txtfile.write(f"Mean Max Feret: {max_feret_mean:.3f} ± {max_feret_sd:.3f} mm (SD, n={n}), SEM = {max_feret_sem:.3f} \n")
    txtfile.write(f"Mean Min Feret: {min_feret_mean:.3f} ± {min_feret_sd:.3f} mm (SD, n={n}), SEM = {min_feret_sem:.3f} \n")
    txtfile.write(f"Average contour max Feret: {max_feret_point:.3f} mm, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")

print("Average ferets exported to average_ferets {}.txt".format(VIDEO_ID))

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

