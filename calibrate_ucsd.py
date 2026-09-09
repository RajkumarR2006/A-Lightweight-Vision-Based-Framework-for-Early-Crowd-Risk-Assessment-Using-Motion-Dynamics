import os
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2"
TRAIN_ROOT = os.path.join(DATASET_ROOT, "UCSDped1", "Train")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "ucsd_calibration.txt")

DENSITY_LOW_PERCENTILE = 5
DENSITY_HIGH_PERCENTILE = 95
MII_LOW_PERCENTILE = 5
MII_HIGH_PERCENTILE = 95

if not os.path.exists(TRAIN_ROOT):
    raise FileNotFoundError(f"Training directory not found:\n{TRAIN_ROOT}")

sequences = sorted([
    name for name in os.listdir(TRAIN_ROOT)
    if os.path.isdir(os.path.join(TRAIN_ROOT, name))
    and name.lower().startswith("train")
])

print("========================================")
print("CROWD SENSE")
print("UCSD CALIBRATION")
print("========================================")
print("Training directory:")
print(TRAIN_ROOT)
print()
print("Training sequences:", len(sequences))
print()

all_density = []
all_mii = []

for sequence in sequences:
    sequence_dir = os.path.join(TRAIN_ROOT, sequence)
    frame_files = sorted([
        f for f in os.listdir(sequence_dir)
        if f.lower().endswith(".tif")
    ])

    if len(frame_files) == 0:
        print(f"{sequence}: no TIFF frames found")
        continue

    print(f"{sequence}: {len(frame_files)} frames")

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )

    previous_frame = None
    kernel = np.ones((3, 3), dtype=np.uint8)

    for frame_file in frame_files:
        frame_path = os.path.join(sequence_dir, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        foreground_mask = background_subtractor.apply(frame)

        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_OPEN, kernel
        )
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_CLOSE, kernel
        )

        density = (
            np.count_nonzero(foreground_mask > 0)
            / foreground_mask.size
        )
        all_density.append(float(density))

        if previous_frame is None:
            previous_frame = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            previous_frame, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, _ = cv2.cartToPolar(
            flow[..., 0], flow[..., 1],
            angleInDegrees=True
        )

        mii = float(np.std(magnitude.astype(np.float32)))
        all_mii.append(mii)

        previous_frame = gray

if len(all_density) == 0:
    raise RuntimeError("No density values were collected.")

if len(all_mii) == 0:
    raise RuntimeError("No MII values were collected.")

density_min = float(np.percentile(all_density, DENSITY_LOW_PERCENTILE))
density_max = float(np.percentile(all_density, DENSITY_HIGH_PERCENTILE))
mii_min = float(np.percentile(all_mii, MII_LOW_PERCENTILE))
mii_max = float(np.percentile(all_mii, MII_HIGH_PERCENTILE))

with open(OUTPUT_FILE, "w") as file:
    file.write(f"DENSITY_MIN={density_min:.8f}\n")
    file.write(f"DENSITY_MAX={density_max:.8f}\n")
    file.write(f"MII_MIN={mii_min:.8f}\n")
    file.write(f"MII_MAX={mii_max:.8f}\n")

print()
print("========================================")
print("CALIBRATION COMPLETE")
print("========================================")
print("Density samples:", len(all_density))
print("MII samples    :", len(all_mii))
print()
print(f"Density 5th percentile : {density_min:.6f}")
print(f"Density 95th percentile: {density_max:.6f}")
print()
print(f"MII 5th percentile     : {mii_min:.6f}")
print(f"MII 95th percentile    : {mii_max:.6f}")
print()
print("Saved to:")
print(OUTPUT_FILE)
print("========================================")
