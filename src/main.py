from pathlib import Path
import sys
import time
import math

import cv2
import numpy as np


# ============================================================
# CROWD SENSE - FINAL DEMO MODE
# Early Crowd Congestion Risk Prediction
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = Path(
    r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2"
)

TRAIN_ROOT = DATASET_ROOT / "UCSDped1" / "Train"

# ------------------------------------------------------------
# Frozen feature calibration learned from UCSD Ped1 TRAIN.
# ------------------------------------------------------------
DENSITY_MIN = 0.00326960
DENSITY_MAX = 0.06023295

MII_MIN = 0.12297923
MII_MAX = 0.46781751

# ------------------------------------------------------------
# Frozen operational thresholds calibrated using normal-only
# Test001 validation sequence.
#
# These values come from risk_thresholds.txt.
# ------------------------------------------------------------
MEDIUM_THRESHOLD = 0.82791673
HIGH_THRESHOLD = 0.87626211

CRI_WINDOW = 5

CPI_WEIGHT = 0.7
ANOMALY_WEIGHT = 0.3

# ------------------------------------------------------------
# Optical flow settings - same as evaluation pipeline.
# ------------------------------------------------------------
FARNEBACK_SETTINGS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

# Background subtraction - same as evaluation pipeline.
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16
MOG2_DETECT_SHADOWS = False


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def normalize(value, minimum, maximum):
    if maximum <= minimum:
        raise ValueError("Maximum must be greater than minimum.")

    return float(
        np.clip(
            (value - minimum) / (maximum - minimum),
            0.0,
            1.0,
        )
    )


def compute_density(mask):
    if mask is None or mask.size == 0:
        raise ValueError("Invalid foreground mask.")

    return float(
        np.count_nonzero(mask > 0) / mask.size
    )


def compute_mii(magnitude):
    if magnitude is None or magnitude.size == 0:
        raise ValueError("Invalid optical-flow magnitude.")

    return float(
        np.std(magnitude.astype(np.float32))
    )


def compute_dci(angle, magnitude):
    """
    Directional Conflict Index.

    Circular variance:
        DCI = 1 - R

    where R is the mean resultant vector length.
    """

    if angle is None or magnitude is None:
        raise ValueError("Angle and magnitude are required.")

    if angle.size == 0 or magnitude.size == 0:
        return 0.0

    threshold = max(
        float(np.percentile(magnitude, 50)),
        1e-6,
    )

    valid = magnitude > threshold

    if not np.any(valid):
        return 0.0

    theta = np.deg2rad(angle[valid])

    mean_cos = float(np.mean(np.cos(theta)))
    mean_sin = float(np.mean(np.sin(theta)))

    resultant_length = math.sqrt(
        mean_cos ** 2 + mean_sin ** 2
    )

    return float(
        np.clip(
            1.0 - resultant_length,
            0.0,
            1.0,
        )
    )


# ============================================================
# NORMAL BEHAVIOUR MODEL
# ============================================================

def extract_training_features(sequence_dir):
    """
    Extract normal-behaviour features from one UCSD Ped1
    training sequence.
    """

    frame_files = sorted(
        Path(sequence_dir).glob("*.tif")
    )

    if not frame_files:
        raise FileNotFoundError(
            f"No .tif frames found in {sequence_dir}"
        )

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )

    kernel = np.ones((3, 3), dtype=np.uint8)

    previous_gray = None

    density_values = []
    mii_values = []
    dci_values = []

    for frame_file in frame_files:

        frame = cv2.imread(str(frame_file))

        if frame is None:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        mask = subtractor.apply(frame)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        d = compute_density(mask)

        if previous_gray is None:
            previous_gray = gray
            density_values.append(d)
            continue

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            FARNEBACK_SETTINGS["pyr_scale"],
            FARNEBACK_SETTINGS["levels"],
            FARNEBACK_SETTINGS["winsize"],
            FARNEBACK_SETTINGS["iterations"],
            FARNEBACK_SETTINGS["poly_n"],
            FARNEBACK_SETTINGS["poly_sigma"],
            FARNEBACK_SETTINGS["flags"],
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
            angleInDegrees=True,
        )

        density_values.append(d)
        mii_values.append(
            compute_mii(magnitude)
        )
        dci_values.append(
            compute_dci(angle, magnitude)
        )

        previous_gray = gray

    return (
        np.asarray(density_values, dtype=np.float32),
        np.asarray(mii_values, dtype=np.float32),
        np.asarray(dci_values, dtype=np.float32),
    )


def learn_normal_model():
    """
    Learn median and robust scale from UCSD Ped1 TRAIN only.
    """

    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(
            f"Training directory not found:\n{TRAIN_ROOT}"
        )

    density_all = []
    mii_all = []
    dci_all = []

    sequences = sorted(
        p for p in TRAIN_ROOT.iterdir()
        if p.is_dir()
    )

    if not sequences:
        raise RuntimeError(
            "No UCSD Ped1 training sequences found."
        )

    print(
        f"Learning normal behaviour from "
        f"{len(sequences)} training sequences..."
    )

    for sequence in sequences:

        print(
            f"  Processing {sequence.name}..."
        )

        density_values, mii_values, dci_values = (
            extract_training_features(sequence)
        )

        density_all.append(density_values)

        if len(mii_values) > 0:
            mii_all.append(mii_values)

        if len(dci_values) > 0:
            dci_all.append(dci_values)

    density_all = np.concatenate(density_all)
    mii_all = np.concatenate(mii_all)
    dci_all = np.concatenate(dci_all)

    statistics = {}

    for name, values in [
        ("density", density_all),
        ("mii", mii_all),
        ("dci", dci_all),
    ]:

        median = float(
            np.median(values)
        )

        mad = float(
            np.median(
                np.abs(values - median)
            )
        )

        scale = 1.4826 * mad

        if scale < 1e-6:
            scale = float(
                np.std(values)
            )

        scale = max(
            scale,
            1e-6,
        )

        statistics[name] = (
            median,
            scale,
        )

    return statistics


# ============================================================
# CROWD RISK CALCULATION
# ============================================================

def compute_risk(
    density,
    motion_instability,
    directional_conflict,
    statistics,
):
    density_n = normalize(
        density,
        DENSITY_MIN,
        DENSITY_MAX,
    )

    mii_n = normalize(
        motion_instability,
        MII_MIN,
        MII_MAX,
    )

    dci_n = float(
        np.clip(
            directional_conflict,
            0.0,
            1.0,
        )
    )

    # Crowd Pressure Index.
    cpi = (
        density_n
        * (
            0.5 * mii_n
            + 0.5 * dci_n
        )
    )

    # Deviation from learned normal behaviour.
    deviations = []

    for feature, value in [
        ("density", density),
        ("mii", motion_instability),
        ("dci", directional_conflict),
    ]:

        median, scale = statistics[feature]

        z = abs(value - median) / scale

        deviations.append(z)

    mean_abs_z = float(
        np.mean(deviations)
    )

    anomaly = (
        1.0
        - np.exp(
            -mean_abs_z / 2.0
        )
    )

    combined_risk = (
        CPI_WEIGHT * cpi
        + ANOMALY_WEIGHT * anomaly
    )

    combined_risk = float(
        np.clip(
            combined_risk,
            0.0,
            1.0,
        )
    )

    return (
        density_n,
        mii_n,
        dci_n,
        float(cpi),
        float(anomaly),
        combined_risk,
    )


# ============================================================
# INPUT HANDLING
# ============================================================

def get_frame_files(input_path):
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input does not exist:\n{path}"
        )

    if path.is_dir():

        files = sorted(
            list(path.glob("*.tif"))
            + list(path.glob("*.jpg"))
            + list(path.glob("*.jpeg"))
            + list(path.glob("*.png"))
        )

        if not files:
            raise FileNotFoundError(
                f"No supported image frames found in:\n{path}"
            )

        return files

    return None


def draw_risk_bar(frame, risk):
    height, width = frame.shape[:2]

    x1 = width - 60
    x2 = width - 35

    top = 170
    bottom = height - 80

    cv2.rectangle(
        frame,
        (x1, top),
        (x2, bottom),
        (255, 255, 255),
        2,
    )

    filled_height = int(
        (bottom - top) * risk
    )

    if filled_height > 0:

        cv2.rectangle(
            frame,
            (
                x1 + 2,
                bottom - filled_height,
            ),
            (
                x2 - 2,
                bottom - 2,
            ),
            (0, 0, 255),
            -1,
        )


def put_text(frame, text, position, scale=0.65, thickness=2):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# DEMO
# ============================================================

def run_demo(input_path):

    frame_files = get_frame_files(input_path)

    if frame_files is not None:

        mode = "images"

        capture = None

        print(
            f"Input mode: image sequence"
        )

        def get_frame():
            nonlocal frame_index

            if frame_index >= len(frame_files):
                return None

            file = frame_files[frame_index]
            frame_index += 1

            return cv2.imread(str(file))

        frame_index = 0

    else:

        mode = "video"

        capture = cv2.VideoCapture(
            str(input_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video:\n{input_path}"
            )

        print(
            "Input mode: video"
        )

        def get_frame():
            ok, frame = capture.read()

            if not ok:
                return None

            return frame

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    previous_gray = None

    cri_history = []

    fps_history = []

    frame_number = 0

    print("\nStarting CrowdSense...")
    print("Press Q to quit.")
    print("Image-sequence playback: approximately 12.5 FPS.")
    print(
        "Prediction logic: CRI compared with "
        "frozen normal-validation thresholds."
    )

    while True:

        frame = get_frame()

        if frame is None:
            break

        frame_number += 1

        start_time = time.perf_counter()

        # UCSD frames are only 238x158, so enlarge them for a clear
        # final-review display.
        display = cv2.resize(
            frame,
            None,
            fx=2.5,
            fy=2.5,
            interpolation=cv2.INTER_NEAREST,
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        mask = subtractor.apply(frame)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        density = compute_density(mask)

        if previous_gray is None:

            previous_gray = gray

            mii_value = 0.0
            dci_value = 0.0

        else:

            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                FARNEBACK_SETTINGS["pyr_scale"],
                FARNEBACK_SETTINGS["levels"],
                FARNEBACK_SETTINGS["winsize"],
                FARNEBACK_SETTINGS["iterations"],
                FARNEBACK_SETTINGS["poly_n"],
                FARNEBACK_SETTINGS["poly_sigma"],
                FARNEBACK_SETTINGS["flags"],
            )

            magnitude, angle = cv2.cartToPolar(
                flow[..., 0],
                flow[..., 1],
                angleInDegrees=True,
            )

            mii_value = compute_mii(
                magnitude
            )

            dci_value = compute_dci(
                angle,
                magnitude,
            )

            # Sparse optical-flow visualization.
            step = 20

            for y in range(
                0,
                flow.shape[0],
                step,
            ):

                for x in range(
                    0,
                    flow.shape[1],
                    step,
                ):

                    fx = float(
                        flow[y, x, 0]
                    )

                    fy = float(
                        flow[y, x, 1]
                    )

                    magnitude_point = math.sqrt(
                        fx * fx + fy * fy
                    )

                    if magnitude_point < 1.0:
                        continue

                    x2 = int(
                        x + fx * 2
                    )

                    y2 = int(
                        y + fy * 2
                    )

                    cv2.arrowedLine(
                        display,
                        (x, y),
                        (x2, y2),
                        (0, 255, 255),
                        1,
                        tipLength=0.3,
                    )

            previous_gray = gray

        (
            density_n,
            mii_n,
            dci_n,
            cpi,
            anomaly,
            combined_risk,
        ) = compute_risk(
            density,
            mii_value,
            dci_value,
            statistics,
        )

        cri_history.append(
            combined_risk
        )

        if len(cri_history) > CRI_WINDOW:
            cri_history.pop(0)

        cri = float(
            np.mean(cri_history)
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )

        fps_history.append(fps)

        # ----------------------------------------------------
        # Prediction state.
        # ----------------------------------------------------
        if cri >= HIGH_THRESHOLD:

            risk_level = "HIGH"

            prediction = (
                "CONGESTION PREDICTED"
            )

            status = (
                "EARLY WARNING ACTIVE"
            )

        elif cri >= MEDIUM_THRESHOLD:

            risk_level = "MEDIUM"

            prediction = (
                "CONGESTION DEVELOPING"
            )

            status = (
                "MONITOR CLOSELY"
            )

        else:

            risk_level = "LOW"

            prediction = (
                "NO CONGESTION PREDICTED"
            )

            status = (
                "NORMAL CROWD"
            )

        # ----------------------------------------------------
        # Visualization panel.
        # ----------------------------------------------------
        height, width = display.shape[:2]

        panel_width = 390

        panel = display[
            :,
            max(0, width - panel_width):width,
        ]

        overlay = panel.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                panel.shape[1],
                panel.shape[0],
            ),
            (0, 0, 0),
            -1,
        )

        display[
            :,
            max(0, width - panel_width):width,
        ] = cv2.addWeighted(
            panel,
            0.25,
            overlay,
            0.75,
            0,
        )

        panel_x = max(
            20,
            width - panel_width + 20,
        )

        put_text(
            display,
            "CROWDSENSE",
            (panel_x, 40),
            0.9,
            2,
        )

        put_text(
            display,
            "EARLY CONGESTION PREDICTION",
            (panel_x, 68),
            0.45,
            1,
        )

        put_text(
            display,
            f"Density : {density:.4f}",
            (panel_x, 110),
        )

        put_text(
            display,
            f"MII     : {mii_value:.4f}",
            (panel_x, 140),
        )

        put_text(
            display,
            f"DCI     : {dci_value:.4f}",
            (panel_x, 170),
        )

        put_text(
            display,
            f"CPI     : {cpi:.4f}",
            (panel_x, 200),
        )

        put_text(
            display,
            f"Anomaly : {anomaly:.4f}",
            (panel_x, 230),
        )

        put_text(
            display,
            f"Risk    : {combined_risk:.4f}",
            (panel_x, 260),
        )

        put_text(
            display,
            f"CRI     : {cri:.4f}",
            (panel_x, 290),
        )

        put_text(
            display,
            f"LEVEL   : {risk_level}",
            (panel_x, 330),
        )

        put_text(
            display,
            prediction,
            (panel_x, 365),
            0.52,
            2,
        )

        put_text(
            display,
            status,
            (panel_x, 395),
            0.48,
            1,
        )

        put_text(
            display,
            f"FPS     : {fps:.1f}",
            (panel_x, 435),
        )

        put_text(
            display,
            f"Frame   : {frame_number}",
            (panel_x, 465),
        )

        # Risk meter.
        draw_risk_bar(
            display,
            cri,
        )

        # Threshold labels.
        put_text(
            display,
            f"M {MEDIUM_THRESHOLD:.2f}",
            (
                width - 145,
                height - 125,
            ),
            0.42,
            1,
        )

        put_text(
            display,
            f"H {HIGH_THRESHOLD:.2f}",
            (
                width - 145,
                height - 95,
            ),
            0.42,
            1,
        )

        cv2.imshow(
            "CrowdSense - Early Congestion Prediction",
            display,
        )

        # Image sequences need a visible playback delay.
        # Videos use a short delay so they remain close to real-time.
        delay_ms = 80 if mode == "images" else 1

        key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord("q"):
            break

    if capture is not None:
        capture.release()

    # Keep the final frame visible for the final review.
    print("Demo complete. Press any key in the CrowdSense window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    average_fps = (
        float(np.mean(fps_history))
        if fps_history
        else 0.0
    )

    print("\nDemo finished.")
    print(
        f"Average processing FPS: "
        f"{average_fps:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            'python main.py "<video_or_frame_directory>"'
        )

        print(
            "\nExample normal:"
        )

        print(
            r'python main.py "E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test001"'
        )

        print(
            "\nExample congestion/anomaly:"
        )

        print(
            r'python main.py "E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Test\Test003"'
        )

        sys.exit(1)

    statistics = learn_normal_model()

    print("\nNormal model learned.")

    for feature, (median, scale) in statistics.items():

        print(
            f"{feature:8s}: "
            f"median={median:.6f}, "
            f"scale={scale:.6f}"
        )

    run_demo(
        sys.argv[1]
    )
