import os
import cv2
import numpy as np

from mii import compute_mii
from dci import compute_dci


# --------------------------------------------------
# Dataset Path
# --------------------------------------------------
DATASET_PATH = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test001"
# --------------------------------------------------
# Read all TIFF frames
# --------------------------------------------------
frame_files = sorted(
    [f for f in os.listdir(DATASET_PATH) if f.endswith(".tif")]
)

# --------------------------------------------------
# Read First Frame
# --------------------------------------------------
prev_frame = cv2.imread(
    os.path.join(DATASET_PATH, frame_files[0]),
    cv2.IMREAD_GRAYSCALE
)

# --------------------------------------------------
# Process Remaining Frames
# --------------------------------------------------
for i in range(1, len(frame_files)):

    current_frame = cv2.imread(
        os.path.join(DATASET_PATH, frame_files[i]),
        cv2.IMREAD_GRAYSCALE
    )

    # ----------------------------------------------
    # Dense Optical Flow (Farneback)
    # ----------------------------------------------
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

    # ----------------------------------------------
    # Magnitude and Direction
    # ----------------------------------------------
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=True
    )

    # ----------------------------------------------
    # Basic Motion Statistics
    # ----------------------------------------------
    avg_magnitude = np.mean(magnitude)
    avg_direction = np.mean(angle)

    # ----------------------------------------------
    # Motion Instability Index
    # ----------------------------------------------
    mii = compute_mii(magnitude)

    # ----------------------------------------------
    # Direction Conflict Index
    # ----------------------------------------------
    dci = compute_dci(angle, magnitude)

    # ----------------------------------------------
    # Display Results
    # ----------------------------------------------
    print("=" * 60)
    print(f"Frame                  : {frame_files[i]}")
    print(f"Average Magnitude      : {avg_magnitude:.4f}")
    print(f"Average Direction      : {avg_direction:.2f}°")
    print(f"Motion Instability     : {mii:.4f}")
    print(f"Direction Conflict     : {dci:.4f}")
    print("=" * 60)

    # ----------------------------------------------
    # Show Current Frame
    # ----------------------------------------------
    cv2.imshow("Current Frame", current_frame)

    key = cv2.waitKey(30)

    if key == 27:      # ESC key
        break

    prev_frame = current_frame.copy()

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cv2.destroyAllWindows()