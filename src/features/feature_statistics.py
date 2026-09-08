import os
import cv2
import numpy as np

from mii import compute_mii
from dci import compute_dci
from density import compute_density


# --------------------------------------------------
# Dataset Path
# --------------------------------------------------
DATASET_PATH = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test001"


# --------------------------------------------------
# Read Frames
# --------------------------------------------------
frame_files = sorted(
    [f for f in os.listdir(DATASET_PATH) if f.endswith(".tif")]
)

if len(frame_files) < 2:
    raise ValueError("Not enough frames found.")


# --------------------------------------------------
# Background Subtractor
# --------------------------------------------------
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)


# --------------------------------------------------
# Storage
# --------------------------------------------------
density_values = []
mii_values = []
dci_values = []
magnitude_values = []


# --------------------------------------------------
# First Frame
# --------------------------------------------------
prev_frame = cv2.imread(
    os.path.join(DATASET_PATH, frame_files[0]),
    cv2.IMREAD_GRAYSCALE
)


# --------------------------------------------------
# Process Frames
# --------------------------------------------------
for i in range(1, len(frame_files)):

    current_frame = cv2.imread(
        os.path.join(DATASET_PATH, frame_files[i]),
        cv2.IMREAD_GRAYSCALE
    )

    # Background subtraction
    fg_mask = background_subtractor.apply(current_frame)

    # Density
    density = compute_density(fg_mask)

    # Optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame,
        current_frame,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    # Magnitude and direction
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=True
    )

    # Features
    avg_magnitude = np.mean(magnitude)
    mii = compute_mii(magnitude)
    dci = compute_dci(angle, magnitude)

    # Store
    density_values.append(density)
    mii_values.append(mii)
    dci_values.append(dci)
    magnitude_values.append(avg_magnitude)

    prev_frame = current_frame.copy()


# --------------------------------------------------
# Statistics
# --------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE STATISTICS")
print("=" * 60)

print("\nDensity")
print(f"Min    : {np.min(density_values):.4f}")
print(f"Max    : {np.max(density_values):.4f}")
print(f"Mean   : {np.mean(density_values):.4f}")
print(f"Std    : {np.std(density_values):.4f}")

print("\nMII")
print(f"Min    : {np.min(mii_values):.4f}")
print(f"Max    : {np.max(mii_values):.4f}")
print(f"Mean   : {np.mean(mii_values):.4f}")
print(f"Std    : {np.std(mii_values):.4f}")

print("\nDCI")
print(f"Min    : {np.min(dci_values):.4f}")
print(f"Max    : {np.max(dci_values):.4f}")
print(f"Mean   : {np.mean(dci_values):.4f}")
print(f"Std    : {np.std(dci_values):.4f}")

print("\nAverage Motion Magnitude")
print(f"Min    : {np.min(magnitude_values):.4f}")
print(f"Max    : {np.max(magnitude_values):.4f}")
print(f"Mean   : {np.mean(magnitude_values):.4f}")
print(f"Std    : {np.std(magnitude_values):.4f}")

print("\n" + "=" * 60)