# Extract Single MU

Detects single motor unit twitches in a ROI of an ultrasound video and extracts Feret diameters (with confidence intervals) of an averaged motor unit contour. Option to synchronise to known electrical stimulus timing, or run over the whole video.

## Overview

The pipeline takes:
- An ultrasound video of a twitching muscle.
- A JSON of motion chunks corresponding to the video (`motion_chunks.json`), from which EMG stimulus times can be extracted

And produces:
- A per-stimulus detection of the motor unit's active (twitching) region
- An averaged contour across all detected twitches, with a consensus heatmap
- A TXT of max/min Feret diameters of the averaged contour, with bootstrap confidence
  intervals
- A PNG showing the heatmap with average contour overlaid on the first frame of the video
- A CSV of per-twitch areas and Feret measurements

## Requirements

```
opencv-python
numpy
scipy
```
The scripts in this package are intended to be run after after the MUNE pipeline. 

## Pipeline stages

### ROI selection and pixel calibration

#### Usage

```bash
python calculate_roi_pixels.py
```

1. **US field ROI selected** - a GUI is used to first select the whole US frame, defining the region of interest (ROI) (`x0, y0, w, h`) of the US window
2. **Pixel dimensions calculated** - the dimensions of pixels in the video are then calculated using the US ROI and known width and depth of the US video
3. **Motor unit ROI selected** - a second GUI is then used to define a fixed ROI in which to look for the muscle twitch (`x0, y0, w, h`), assumed static across the recording. The user should bound the minimum area perceived to contain the twitching motor unit, based on previously reviewing the video.
4. **Parameters exported**  - A JSON file is created, containing dictionaries of the ROIs of the US field (`roi_US`) and motor unit area (`roi_MU`) and the pixel dimensions (`roi_pixels`).

#### User Parameters

Set these at the top of the script before running:

| Parameter | Description |
|---|---|
| `VIDEO_ID` | ID of the video, used in naming output files |
| `VIDEO_PATH` | Path to the ultrasound video file |
| `width` | Width of US video field (mm) |
| `depth` | Depth of US video field (mm) |

#### Output Files

| File | Description |
|---|---|
| `roi_pixels {VIDEO_ID}.json` | JSON file containing dictionaries with pixel dimensions and ROIs for the video |

### Motor Unit Identification

#### Usage

```bash
python extract_motor_unit.py
```

1. **ROI and pixel dimensions loaded** - ROI of the motor unit and pixel dimensions for the video loaded from `roi_pixels {VIDEO_ID}.json`. If no JSON file present, script with exit and prompt user to run `calculate_roi_pixels.py`.
2. **Baseline noise calculated** - variance of early frames of the video are sampled to estimate the baseline noise.
3. **Pre-processing** —  Frames are read, filtered using a Gaussian blur, cropped to the ROI, and pushed through a rolling buffer. If `LOCK_TO_STIM_TIMES = True`, stimulus times are first loaded from `motion_chunks.json` and for each known stimulus time, a search window is opened starting after an expected electromechanical delay. In this case, only frames in the search window are processed.
4. **Temporal variance** — a rolling buffer of `WINDOW_SIZE` frames is used to compute a per-pixel variance map. If the median and standard deviation of the baseline noise are greater than pre-defined values, a temporal bandpass filter is first applied to the variance map, which is then Gaussian blurred and passed through a top-hat filter. If noise is low then only Gaussian blurring is used. High variance indicates motion (twitch), low variance indicates a static background.
5. **Thresholding** — a robust median + MAD threshold separates active (twitching) pixels from background noise within the variance map. Scale factor _k_ is adaptive based on baseline noise of the video.
6. **Confirmation** — a twitch is only confirmed once the thresholded active area exceeds `AREA_RATIO_THRESH` of the ROI for `MIN_DETECTIONS` out of the last `CONFIRM_WINDOW` frames. This avoids single-frame noise spikes being registered as twitches.
7. **Peak frame extraction** — once a twitch is confirmed, the variance map with the highest total energy during that trial is kept as the representative frame for that twitch.
8. **Contour extraction** — the peak variance map is thresholded, cleaned with morphological opening/closing, and the largest connected component is converted to a contour (convex hull).
9. **Per-video averaging** — contours from all confirmed twitches in a video are rasterised into binary masks and averaged pixel-wise to produce a consensus probability map (`prob_mask`) and a thresholded average mask.
10. **Feret diameter extraction** — maximum Feret diameter and minimum Feret diameter are computed on the averaged contour and scaled to physical units (mm) using pixel calibration.
11. **Error estimation** — a bootstrap resampling of the per-twitch masks (resample with replacement, rebuild the average contour, recompute Feret) gives a 95% confidence interval on the averaged contour's Feret diameters. Per-twitch mean ± standard deviation and standard error on the mean is also reported directly from individual twitch measurements. These values, along with the average contour's max Feret are exported as a TXT file.
12. **Visualisation** — the consensus heatmap and average contour outline, calipers are overlaid on the first frame of the video and exported as a PNG.

#### User Parameters

Set these at the top of the script before running:

| Parameter | Description |
|---|---|
| `VIDEO_ID` | ID of the video, used in naming output files |
| `VIDEO_PATH` | Path to the ultrasound video file |
| `FPS` | Video frame rate. Sampling rate in `motion_metrics.json`|
| `WINDOW_SIZE` | Number of frames in the rolling buffer used to compute temporal variance. Should approximate the expected twitch duration in frames (see [Tuning](#tuning)) |
| `CONFIRM_WINDOW` | Number of recent frames checked for sustained detection before confirming a twitch |
| `MIN_DETECTIONS` | Minimum number of frames within `CONFIRM_WINDOW` that must exceed the area threshold to confirm a twitch |
| `AREA_RATIO_THRESH` | Fraction of ROI area that must be "active" (above the MAD threshold) to count as motion in a given frame |
| `MIN_AREA_PIXELS` | Minimum connected pixel count for a detected region to be considered a real motor unit, not noise |
| `LOCK_TO_STIM_TIMES` | Boolean. Defines whether pipeline runs off whole video of search windows locked to known stimulus times |
| `delay_min_sec` | Defines how long to wait after a stimulus to start looking for twitch motion |
| `delay_max_sec` | Defines maximum time after stimulus to look for twitch motion|

#### Output Files

| File | Description |
|---|---|
| `twitch_areas {VIDEO_ID}.csv` | CSV file of the start time (_s_), end time (_s_), motor unit area area (_mm<sup>2</sup>_), minimum Feret diameter (_mm_) and maximum Feret diameter (_mm_) of all captured twitches. |
| `average_ferets {VIDE0_ID}.txt` | TXT file containing the average maximum and minimum Feret diameters (_mm_) across all twitches with standard deviation (SD) and standard error on the mean (SEM), and the average contour's maximum Feret diameter (_mm_)|
| `average_motor_unit_heatmap {VIDEO_ID}.png` | Heatmap of the probability mask of the motor unit based on locations of all recorded twitches with the average contour projected onto the first frame of the video. |
