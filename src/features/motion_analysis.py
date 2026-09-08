import os
import cv2
import numpy as np

from mii import compute_mii
from dci import compute_dci
from density import compute_density
from cpi import compute_cpi
from cri import compute_cri


# ==================================================
# Dataset Path
# ==================================================

DATASET_PATH = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test001"


# ==================================================
# CPI Normalization Parameters
# ==================================================

DENSITY_MIN = 0.0025
DENSITY_MAX = 0.0617

MII_MIN = 0.2298
MII_MAX = 0.6886


# ==================================================
# CRI Parameters
# ==================================================

CRI_WINDOW = 5


# ==================================================
# Read TIFF Frames
# ==================================================

frame_files = sorted(
    [
        f for f in os.listdir(DATASET_PATH)
        if f.lower().endswith(".tif")
    ]
)

if len(frame_files) < 2:
    raise ValueError("Not enough frames found.")


# ==================================================
# Background Subtractor
# ==================================================

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)


# ==================================================
# Storage
# ==================================================

density_values = []
mii_values = []
dci_values = []
magnitude_values = []
cpi_values = []
cri_values = []


# ==================================================
# Read First Frame
# ==================================================

prev_frame = cv2.imread(
    os.path.join(DATASET_PATH, frame_files[0]),
    cv2.IMREAD_GRAYSCALE
)

if prev_frame is None:
    raise ValueError(
        f"Could not read frame: {frame_files[0]}"
    )


# ==================================================
# Process Frames
# ==================================================

for i in range(1, len(frame_files)):

    # ------------------------------------------------
    # Read Current Frame
    # ------------------------------------------------

    current_frame = cv2.imread(
        os.path.join(DATASET_PATH, frame_files[i]),
        cv2.IMREAD_GRAYSCALE
    )

    if current_frame is None:
        print(f"Skipping unreadable frame: {frame_files[i]}")
        continue


    # =================================================
    # Background Subtraction
    # =================================================

    fg_mask = background_subtractor.apply(current_frame)


    # =================================================
    # Morphological Cleanup
    # =================================================

    kernel = np.ones((3, 3), np.uint8)

    fg_mask = cv2.morphologyEx(
        fg_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    fg_mask = cv2.morphologyEx(
        fg_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # =================================================
    # Density
    # =================================================

    density = compute_density(fg_mask)


    # =================================================
    # Optical Flow
    # =================================================

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


    # =================================================
    # Magnitude and Direction
    # =================================================

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=True
    )


    # =================================================
    # Basic Motion Statistics
    # =================================================

    avg_magnitude = np.mean(magnitude)
    avg_direction = np.mean(angle)


    # =================================================
    # MII
    # =================================================

    mii = compute_mii(magnitude)


    # =================================================
    # DCI
    # =================================================

    dci = compute_dci(angle, magnitude)


    # =================================================
    # CPI
    # =================================================

    cpi = compute_cpi(
        density=density,
        mii=mii,
        dci=dci,
        density_min=DENSITY_MIN,
        density_max=DENSITY_MAX,
        mii_min=MII_MIN,
        mii_max=MII_MAX
    )


    # =================================================
    # Store CPI
    # =================================================

    cpi_values.append(cpi)


    # =================================================
    # CRI
    # =================================================

    cri = compute_cri(
        cpi_values,
        window_size=CRI_WINDOW
    )


    cri_values.append(cri)


    # =================================================
    # Store Other Features
    # =================================================

    density_values.append(density)
    mii_values.append(mii)
    dci_values.append(dci)
    magnitude_values.append(avg_magnitude)


    # =================================================
    # Display
    # =================================================

    print("=" * 60)

    print(f"Frame                  : {frame_files[i]}")
    print(f"Average Magnitude      : {avg_magnitude:.4f}")
    print(f"Average Direction      : {avg_direction:.2f}°")
    print(f"Motion Instability     : {mii:.4f}")
    print(f"Direction Conflict     : {dci:.4f}")
    print(f"Crowd Density          : {density:.4f}")
    print(f"Congestion Pressure    : {cpi:.4f}")
    print(f"Crowd Risk Index       : {cri:.4f}")

    print("=" * 60)


    # =================================================
    # Update Previous Frame
    # =================================================

    prev_frame = current_frame.copy()


# ==================================================
# Convert to NumPy Arrays
# ==================================================

density_values = np.array(density_values)
mii_values = np.array(mii_values)
dci_values = np.array(dci_values)
magnitude_values = np.array(magnitude_values)
cpi_values = np.array(cpi_values)
cri_values = np.array(cri_values)


# ==================================================
# Final Statistics
# ==================================================

print("\n")
print("=" * 60)
print("FINAL FEATURE STATISTICS")
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


print("\nCPI")
print(f"Min    : {np.min(cpi_values):.4f}")
print(f"Max    : {np.max(cpi_values):.4f}")
print(f"Mean   : {np.mean(cpi_values):.4f}")
print(f"Std    : {np.std(cpi_values):.4f}")


print("\nCRI")
print(f"Min    : {np.min(cri_values):.4f}")
print(f"Max    : {np.max(cri_values):.4f}")
print(f"Mean   : {np.mean(cri_values):.4f}")
print(f"Std    : {np.std(cri_values):.4f}")


print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)