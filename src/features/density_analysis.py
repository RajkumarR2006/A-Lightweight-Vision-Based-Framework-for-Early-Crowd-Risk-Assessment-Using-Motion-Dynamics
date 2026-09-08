import os
import cv2
import numpy as np

from density import compute_density


DATASET_PATH = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test001"


frame_files = sorted(
    [f for f in os.listdir(DATASET_PATH) if f.endswith(".tif")]
)

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)

density_values = []


# --------------------------------------------------
# Process Frames
# --------------------------------------------------
for i, filename in enumerate(frame_files):

    frame = cv2.imread(
        os.path.join(DATASET_PATH, filename),
        cv2.IMREAD_GRAYSCALE
    )

    fg_mask = background_subtractor.apply(frame)

    # Morphological cleanup
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

    density = compute_density(fg_mask)

    density_values.append(density)


# --------------------------------------------------
# Statistics
# --------------------------------------------------
density_values = np.array(density_values)

print("=" * 60)
print("IMPROVED DENSITY STATISTICS")
print("=" * 60)

print(f"Min       : {np.min(density_values):.4f}")
print(f"Max       : {np.max(density_values):.4f}")
print(f"Mean      : {np.mean(density_values):.4f}")
print(f"Std       : {np.std(density_values):.4f}")

print("\nPercentiles:")

print(f"5th       : {np.percentile(density_values, 5):.4f}")
print(f"25th      : {np.percentile(density_values, 25):.4f}")
print(f"50th      : {np.percentile(density_values, 50):.4f}")
print(f"75th      : {np.percentile(density_values, 75):.4f}")
print(f"95th      : {np.percentile(density_values, 95):.4f}")

print("=" * 60)