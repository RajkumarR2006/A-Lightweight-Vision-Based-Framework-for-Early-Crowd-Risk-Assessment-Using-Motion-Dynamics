"""
motion_analysis.py
----------------------------------------------------
Motion Analysis Module

This module:
1. Reads video frames from the UCSD dataset.
2. Computes Dense Optical Flow using Farneback.
3. Calculates Motion Magnitude.
4. Calculates Motion Direction.
5. Computes Motion Instability Index (MII).
"""

import cv2
import numpy as np
import os

from mii import compute_mii


# --------------------------------------------------
# Dataset Path
# --------------------------------------------------

sequence_path = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Train\Train001"

# Read all image filenames
image_files = sorted(os.listdir(sequence_path))

if len(image_files) == 0:
    raise ValueError("No images found in the dataset folder.")

# --------------------------------------------------
# Read First Frame
# --------------------------------------------------

first_frame = cv2.imread(os.path.join(sequence_path, image_files[0]))

if first_frame is None:
    raise ValueError("Unable to read the first image.")

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# Process Remaining Frames
# --------------------------------------------------

for image_name in image_files[1:]:

    frame_path = os.path.join(sequence_path, image_name)

    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------------------------------
    # Dense Optical Flow
    # ----------------------------------------------

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # ----------------------------------------------
    # Motion Magnitude and Direction
    # ----------------------------------------------

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=True
    )

    # ----------------------------------------------
    # Motion Statistics
    # ----------------------------------------------

    average_magnitude = np.mean(magnitude)
    average_direction = np.mean(angle)

    # ----------------------------------------------
    # Motion Instability Index
    # ----------------------------------------------

    mii = compute_mii(magnitude)

    # ----------------------------------------------
    # Display Results
    # ----------------------------------------------

    print("=" * 50)
    print(f"Frame                  : {image_name}")
    print(f"Average Magnitude      : {average_magnitude:.4f}")
    print(f"Average Direction      : {average_direction:.2f}°")
    print(f"Motion Instability     : {mii:.4f}")

    # ----------------------------------------------
    # Display Frame
    # ----------------------------------------------

    cv2.imshow("CrowdSense - Motion Analysis", frame)

    prev_gray = gray

    key = cv2.waitKey(30)

    if key == ord('q'):
        break

cv2.destroyAllWindows()