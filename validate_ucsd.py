import sys
import os
import cv2
import numpy as np
import csv


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.append(SRC_DIR)


# ============================================================
# CROWD SENSE FEATURES
# ============================================================

from features.mii import compute_mii
from features.dci import compute_dci
from features.density import compute_density
from features.cpi import compute_cpi
from features.cri import compute_cri


# ============================================================
# DATASET PATH
# ============================================================

DATASET_ROOT = (
    r"E:\PROJECTS\OPEN_CV\datasets"
    r"\UCSD_Anomaly_Dataset.v1p2"
)

VIDEO_DIR = os.path.join(
    DATASET_ROOT,
    "UCSDped1",
    "Test",
    "Test003"
)

GT_DIR = os.path.join(
    DATASET_ROOT,
    "UCSDped1",
    "Test",
    "Test003_gt"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "test003_validation.csv"
)


# ============================================================
# NORMALIZATION PARAMETERS
# ============================================================

DENSITY_MIN = 0.0025
DENSITY_MAX = 0.0617

MII_MIN = 0.2298
MII_MAX = 0.6886


# ============================================================
# CRI
# ============================================================

CRI_WINDOW = 5


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not os.path.exists(VIDEO_DIR):
    raise FileNotFoundError(
        f"Video directory not found:\n{VIDEO_DIR}"
    )

if not os.path.exists(GT_DIR):
    raise FileNotFoundError(
        f"Ground truth directory not found:\n{GT_DIR}"
    )


# ============================================================
# GET TIF VIDEO FRAMES
# ============================================================

frame_files = sorted(
    [
        file_name
        for file_name in os.listdir(VIDEO_DIR)
        if file_name.lower().endswith(".tif")
    ]
)


# ============================================================
# GET GROUND TRUTH
# ============================================================

gt_files = sorted(
    [
        file_name
        for file_name in os.listdir(GT_DIR)
        if file_name.lower().endswith(".bmp")
    ]
)


# ============================================================
# INFORMATION
# ============================================================

print("========================================")
print("UCSD CROWD SENSE VALIDATION")
print("========================================")

print(
    "Video directory :",
    VIDEO_DIR
)

print(
    "Ground truth    :",
    GT_DIR
)

print()

print(
    "Video frames    :",
    len(frame_files)
)

print(
    "GT frames       :",
    len(gt_files)
)

print()


# ============================================================
# VALIDATE FRAME COUNTS
# ============================================================

if len(frame_files) == 0:
    raise RuntimeError(
        "No .tif video frames found."
    )

if len(gt_files) == 0:
    raise RuntimeError(
        "No .bmp ground-truth frames found."
    )


if len(frame_files) != len(gt_files):

    print(
        "WARNING: Video and ground-truth "
        "frame counts are different."
    )


# ============================================================
# BACKGROUND SUBTRACTION
# ============================================================

background_subtractor = (
    cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )
)


# ============================================================
# VARIABLES
# ============================================================

previous_frame = None

cpi_history = []

results = []


# ============================================================
# PROCESS FRAMES
# ============================================================

for index, frame_file in enumerate(frame_files):

    # --------------------------------------------------------
    # VIDEO FRAME
    # --------------------------------------------------------

    frame_path = os.path.join(
        VIDEO_DIR,
        frame_file
    )

    frame = cv2.imread(
        frame_path
    )

    if frame is None:

        print(
            f"WARNING: Could not read {frame_file}"
        )

        continue


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # BACKGROUND SUBTRACTION
    # ========================================================

    foreground_mask = (
        background_subtractor.apply(frame)
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL FILTERING
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # DENSITY
    # ========================================================

    density = compute_density(
        foreground_mask
    )


    # ========================================================
    # GROUND TRUTH
    #
    # 001.tif -> 001.bmp
    # 002.tif -> 002.bmp
    # ========================================================

    frame_number = os.path.splitext(
        frame_file
    )[0]

    gt_file = frame_number + ".bmp"

    gt_path = os.path.join(
        GT_DIR,
        gt_file
    )

    gt_mask = cv2.imread(
        gt_path,
        cv2.IMREAD_GRAYSCALE
    )


    if gt_mask is None:

        print(
            f"WARNING: Ground truth missing for "
            f"{frame_file}"
        )

        ground_truth = 0

    else:

        # Any non-zero pixel means
        # the frame contains an annotated
        # abnormal region.

        ground_truth = int(
            np.count_nonzero(gt_mask) > 0
        )


    # ========================================================
    # FIRST FRAME
    # ========================================================

    if previous_frame is None:

        previous_frame = gray

        results.append(
            [
                index + 1,
                frame_file,
                0.0,
                0.0,
                0.0,
                density,
                0.0,
                0.0,
                0.0,
                ground_truth
            ]
        )

        continue


    # ========================================================
    # OPTICAL FLOW
    # ========================================================

    flow = cv2.calcOpticalFlowFarneback(
        previous_frame,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )


    # ========================================================
    # MAGNITUDE + DIRECTION
    # ========================================================

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
        angleInDegrees=True
    )


    # ========================================================
    # AVERAGE MOTION
    # ========================================================

    avg_magnitude = float(
        np.mean(magnitude)
    )

    avg_direction = float(
        np.mean(angle)
    )


    # ========================================================
    # MII
    # ========================================================

    mii = compute_mii(
        magnitude
    )


    # ========================================================
    # DCI
    # ========================================================

    dci = compute_dci(
        angle,
        magnitude
    )


    # ========================================================
    # CPI
    # ========================================================

    cpi = compute_cpi(
        density=density,
        mii=mii,
        dci=dci,
        density_min=DENSITY_MIN,
        density_max=DENSITY_MAX,
        mii_min=MII_MIN,
        mii_max=MII_MAX,
        motion_weight=0.5,
        direction_weight=0.5
    )


    # ========================================================
    # CRI
    # ========================================================

    cpi_history.append(
        cpi
    )

    cri = compute_cri(
        cpi_history,
        window_size=CRI_WINDOW
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append(
        [
            index + 1,
            frame_file,
            avg_magnitude,
            avg_direction,
            mii,
            density,
            dci,
            cpi,
            cri,
            ground_truth
        ]
    )


    # ========================================================
    # UPDATE PREVIOUS FRAME
    # ========================================================

    previous_frame = gray


    # ========================================================
    # PROGRESS
    # ========================================================

    if (index + 1) % 25 == 0:

        print(
            f"Frame {index + 1:3d} | "
            f"MII={mii:.4f} | "
            f"DCI={dci:.4f} | "
            f"Density={density:.4f} | "
            f"CPI={cpi:.4f} | "
            f"CRI={cri:.4f} | "
            f"GT={ground_truth}"
        )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "frame",
            "filename",
            "avg_magnitude",
            "avg_direction",
            "mii",
            "density",
            "dci",
            "cpi",
            "cri",
            "ground_truth"
        ]
    )

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

ground_truth_values = [
    row[-1]
    for row in results
]

abnormal_frames = sum(
    ground_truth_values
)

normal_frames = (
    len(ground_truth_values)
    - abnormal_frames
)


print()

print("========================================")
print("VALIDATION COMPLETE")
print("========================================")

print(
    "Total frames    :",
    len(results)
)

print(
    "Normal frames   :",
    normal_frames
)

print(
    "Abnormal frames :",
    abnormal_frames
)

print(
    "CSV output      :",
    OUTPUT_FILE
)

print("========================================")