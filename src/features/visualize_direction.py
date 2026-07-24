import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test001"

# -----------------------------
# Read Frames
# -----------------------------
frames = sorted(
    [f for f in os.listdir(DATASET_PATH) if f.endswith(".tif")]
)

# Choose any frame
frame_index = 174

prev = cv2.imread(
    os.path.join(DATASET_PATH, frames[frame_index - 1]),
    cv2.IMREAD_GRAYSCALE
)

curr = cv2.imread(
    os.path.join(DATASET_PATH, frames[frame_index]),
    cv2.IMREAD_GRAYSCALE
)

# -----------------------------
# Optical Flow
# -----------------------------
flow = cv2.calcOpticalFlowFarneback(
    prev,
    curr,
    None,
    0.5,
    3,
    15,
    3,
    5,
    1.2,
    0,
)

magnitude, angle = cv2.cartToPolar(
    flow[..., 0],
    flow[..., 1],
    angleInDegrees=True
)

# -----------------------------
# Remove weak motion
# -----------------------------
threshold = 0.5

mask = magnitude > threshold

filtered_angles = angle[mask]

print("Motion Pixels :", len(filtered_angles))

# -----------------------------
# Plot Histogram
# -----------------------------
plt.figure(figsize=(10,5))

plt.hist(
    filtered_angles,
    bins=36,
    range=(0,360)
)

plt.title("Direction Distribution")
plt.xlabel("Direction (Degrees)")
plt.ylabel("Number of Motion Pixels")

plt.grid(True)

plt.show()