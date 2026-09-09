import sys
import os
import cv2
import numpy as np
import csv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from features.mii import compute_mii
from features.dci import compute_dci
from features.density import compute_density
from features.cpi import compute_cpi
from features.cri import compute_cri


DATASET_ROOT = (
    r"E:\PROJECTS\OPEN_CV\datasets"
    r"\UCSD_Anomaly_Dataset.v1p2"
)

TEST_ROOT = os.path.join(
    DATASET_ROOT,
    "UCSDped1",
    "Test"
)

CALIBRATION_FILE = os.path.join(
    PROJECT_ROOT,
    "ucsd_calibration.txt"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "ucsd_calibrated_validation.csv"
)

CRI_WINDOW = 5
MOTION_WEIGHT = 0.5
DIRECTION_WEIGHT = 0.5

SEQUENCES = [
    "Test003",
    "Test004",
    "Test014",
    "Test018",
    "Test019",
    "Test021",
    "Test022",
    "Test023",
    "Test024",
    "Test032"
]


def load_calibration():
    values = {}

    with open(CALIBRATION_FILE, "r") as file:
        for line in file:
            key, value = line.strip().split("=")
            values[key] = float(value)

    required = [
        "DENSITY_MIN",
        "DENSITY_MAX",
        "MII_MIN",
        "MII_MAX"
    ]

    for key in required:
        if key not in values:
            raise ValueError(
                f"Missing calibration value: {key}"
            )

    return values


def process_sequence(
    sequence_name,
    density_min,
    density_max,
    mii_min,
    mii_max
):

    video_dir = os.path.join(
        TEST_ROOT,
        sequence_name
    )

    gt_dir = os.path.join(
        TEST_ROOT,
        sequence_name + "_gt"
    )

    print()
    print("========================================")
    print(sequence_name)
    print("========================================")

    if not os.path.exists(video_dir):
        print("ERROR: Video directory not found")
        return []

    if not os.path.exists(gt_dir):
        print("ERROR: Ground truth directory not found")
        return []

    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(".tif")
    ])

    gt_files = sorted([
        f for f in os.listdir(gt_dir)
        if f.lower().endswith(".bmp")
    ])

    print("Video frames :", len(frame_files))
    print("GT frames    :", len(gt_files))

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )

    previous_frame = None
    cpi_history = []
    results = []

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    for index, frame_file in enumerate(frame_files):

        frame_path = os.path.join(
            video_dir,
            frame_file
        )

        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"WARNING: Could not read {frame_file}")
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        foreground_mask = background_subtractor.apply(
            frame
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

        density = compute_density(
            foreground_mask
        )

        frame_number = os.path.splitext(
            frame_file
        )[0]

        gt_path = os.path.join(
            gt_dir,
            frame_number + ".bmp"
        )

        gt_mask = cv2.imread(
            gt_path,
            cv2.IMREAD_GRAYSCALE
        )

        if gt_mask is None:
            ground_truth = 0
        else:
            ground_truth = int(
                np.count_nonzero(gt_mask) > 0
            )

        if previous_frame is None:

            previous_frame = gray

            results.append([
                sequence_name,
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
            ])

            continue

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

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
            angleInDegrees=True
        )

        avg_magnitude = float(
            np.mean(magnitude)
        )

        avg_direction = float(
            np.mean(angle)
        )

        mii = compute_mii(
            magnitude
        )

        dci = compute_dci(
            angle,
            magnitude
        )

        cpi = compute_cpi(
            density=density,
            mii=mii,
            dci=dci,
            density_min=density_min,
            density_max=density_max,
            mii_min=mii_min,
            mii_max=mii_max,
            motion_weight=MOTION_WEIGHT,
            direction_weight=DIRECTION_WEIGHT
        )

        cpi_history.append(cpi)

        cri = compute_cri(
            cpi_history,
            window_size=CRI_WINDOW
        )

        results.append([
            sequence_name,
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
        ])

        previous_frame = gray

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

    return results


print("========================================")
print("CROWD SENSE")
print("LEAKAGE-FREE CALIBRATED EVALUATION")
print("========================================")

calibration = load_calibration()

DENSITY_MIN = calibration["DENSITY_MIN"]
DENSITY_MAX = calibration["DENSITY_MAX"]
MII_MIN = calibration["MII_MIN"]
MII_MAX = calibration["MII_MAX"]

print()
print("Frozen calibration:")
print(f"DENSITY_MIN = {DENSITY_MIN:.8f}")
print(f"DENSITY_MAX = {DENSITY_MAX:.8f}")
print(f"MII_MIN     = {MII_MIN:.8f}")
print(f"MII_MAX     = {MII_MAX:.8f}")

all_results = []

for sequence in SEQUENCES:

    sequence_results = process_sequence(
        sequence,
        DENSITY_MIN,
        DENSITY_MAX,
        MII_MIN,
        MII_MAX
    )

    all_results.extend(sequence_results)


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "sequence",
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
    ])

    writer.writerows(all_results)


gt_values = [
    row[-1]
    for row in all_results
]

abnormal_total = sum(gt_values)
normal_total = len(gt_values) - abnormal_total

print()
print("========================================")
print("CALIBRATED EVALUATION COMPLETE")
print("========================================")
print("Sequences processed :", len(SEQUENCES))
print("Total frames        :", len(all_results))
print("Normal frames       :", normal_total)
print("Abnormal frames     :", abnormal_total)
print("Output:")
print(OUTPUT_FILE)
print("========================================")
