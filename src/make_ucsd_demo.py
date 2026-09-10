"""
CrowdSense - UCSD Test003 MP4 Demonstration
Creates a video showing the real frame-by-frame CrowdSense pipeline:
Density -> MII -> DCI -> CPI -> normal-anomaly -> combined risk -> CRI

Run from project root:
    python .\src\make_ucsd_demo.py

The script learns the normal-behaviour model from UCSD Ped1 Train,
then processes Test003 sequentially and writes an annotated MP4.

NOTE:
This is a demonstration of the current CrowdSense pipeline on UCSD.
UCSD abnormal labels are not treated as proof of physical congestion.
"""

import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "UCSD_Anomaly_Dataset.v1p2"
TRAIN_ROOT = DATASET_ROOT / "Train" / "Train001"
# Ped1 training folders are Train001 ... Train034
PED1_ROOT = DATASET_ROOT / "UCSDped1"

TRAIN_PARENT = PED1_ROOT / "Train"
TEST_SEQ = PED1_ROOT / "Test" / "Test003"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "UCSD_Test003_CrowdSense_Demo.mp4"


# ---------------------------------------------------------------------
# Frozen feature calibration used by CrowdSense
# ---------------------------------------------------------------------
DENSITY_MIN = 0.00326960
DENSITY_MAX = 0.06023295
MII_MIN = 0.12297923
MII_MAX = 0.46781751

CRI_WINDOW = 5
CPI_WEIGHT = 0.7
ANOMALY_WEIGHT = 0.3


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize(value, low, high):
    if high <= low:
        return 0.0
    x = (value - low) / (high - low)
    return float(np.clip(x, 0.0, 1.0))


def robust_center_scale(values):
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-6)
    return median, scale


def compute_mii(magnitude):
    return float(np.std(magnitude.astype(np.float32)))


def compute_dci(angle, magnitude):
    """Directional Conflict Index: circular variance of sufficiently
    strong optical-flow vectors."""
    angle = angle.astype(np.float32)
    magnitude = magnitude.astype(np.float32)

    if magnitude.size == 0:
        return 0.0

    # Ignore weak/near-zero flow vectors.
    threshold = float(np.median(magnitude))
    mask = magnitude > max(threshold, 1e-6)

    if np.count_nonzero(mask) < 10:
        mask = magnitude > 1e-6

    if np.count_nonzero(mask) == 0:
        return 0.0

    a = angle[mask]
    mean_cos = float(np.mean(np.cos(a)))
    mean_sin = float(np.mean(np.sin(a)))
    r = np.sqrt(mean_cos ** 2 + mean_sin ** 2)

    return float(np.clip(1.0 - r, 0.0, 1.0))


def compute_cpi(density_n, mii_n, dci):
    return float(density_n * (0.5 * mii_n + 0.5 * dci))


def optical_flow(prev_gray, gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def get_frames(folder):
    files = sorted(
        [p for p in folder.iterdir()
         if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}]
    )
    return files


# ---------------------------------------------------------------------
# Learn normal behaviour from Ped1 Train
# ---------------------------------------------------------------------
def learn_normal_model():
    train_sequences = sorted(
        p for p in TRAIN_PARENT.iterdir()
        if p.is_dir() and p.name.lower().startswith("train")
    )

    if not train_sequences:
        raise FileNotFoundError(
            f"No training sequences found in {TRAIN_PARENT}"
        )

    density_values = []
    mii_values = []
    dci_values = []

    bg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False,
    )

    print(f"Learning normal behaviour from {len(train_sequences)} training sequences...")

    for seq in train_sequences:
        frames = get_frames(seq)
        if len(frames) < 2:
            continue

        prev_gray = None

        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Warm-up/background model.
            fg = bg.apply(frame)

            density = float(np.mean(fg > 0))

            if prev_gray is None:
                prev_gray = gray
                continue

            flow = optical_flow(prev_gray, gray)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Remove very weak motion from feature statistics.
            valid = mag > 0.05

            if np.count_nonzero(valid) < 10:
                mii = 0.0
                dci = 0.0
            else:
                m = mag[valid]
                a = ang[valid]
                mii = compute_mii(m)
                dci = compute_dci(a, m)

            density_values.append(density)
            mii_values.append(mii)
            dci_values.append(dci)

            prev_gray = gray

    if not density_values:
        raise RuntimeError("Could not calculate training features.")

    density_med, density_scale = robust_center_scale(density_values)
    mii_med, mii_scale = robust_center_scale(mii_values)
    dci_med, dci_scale = robust_center_scale(dci_values)

    model = {
        "density_med": density_med,
        "density_scale": density_scale,
        "mii_med": mii_med,
        "mii_scale": mii_scale,
        "dci_med": dci_med,
        "dci_scale": dci_scale,
    }

    print("Normal model learned.")
    print(f"density : median={density_med:.6f}, scale={density_scale:.6f}")
    print(f"mii     : median={mii_med:.6f}, scale={mii_scale:.6f}")
    print(f"dci     : median={dci_med:.6f}, scale={dci_scale:.6f}")

    return model


def normal_anomaly_score(model, density, mii, dci):
    z_density = abs(density - model["density_med"]) / model["density_scale"]
    z_mii = abs(mii - model["mii_med"]) / model["mii_scale"]
    z_dci = abs(dci - model["dci_med"]) / model["dci_scale"]

    mean_abs_z = (z_density + z_mii + z_dci) / 3.0
    return float(1.0 - np.exp(-mean_abs_z / 2.0))


# ---------------------------------------------------------------------
# Video annotation
# ---------------------------------------------------------------------
def put_text(img, text, xy, scale=0.65, thickness=2):
    cv2.putText(
        img,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(frame, values, frame_no, total, fps):
    h, w = frame.shape[:2]

    panel_w = min(430, max(360, w // 3))
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

    # Header
    put_text(panel, "CROWDSENSE", (20, 35), 0.85, 2)
    put_text(panel, "UCSD Ped1 - Test003", (20, 65), 0.55, 1)

    y = 105
    line_gap = 31

    rows = [
        ("Density", values["density"]),
        ("MII", values["mii"]),
        ("DCI", values["dci"]),
        ("CPI", values["cpi"]),
        ("Anomaly", values["anomaly"]),
        ("Combined Risk", values["combined"]),
        ("CRI", values["cri"]),
    ]

    for name, val in rows:
        put_text(panel, f"{name:<14} {val:.4f}", (20, y), 0.56, 1)
        y += line_gap

    y += 12
    put_text(panel, f"Risk: {values['risk']}", (20, y), 0.70, 2)
    y += 38

    put_text(panel, f"Frame: {frame_no} / {total}", (20, y), 0.55, 1)
    y += 30
    put_text(panel, f"Output FPS: {fps:.1f}", (20, y), 0.55, 1)

    y += 45
    put_text(panel, "Pipeline", (20, y), 0.65, 2)
    y += 30
    put_text(panel, "Density + MII + DCI", (20, y), 0.50, 1)
    y += 26
    put_text(panel, "        -> CPI", (20, y), 0.50, 1)
    y += 26
    put_text(panel, "        -> Normal anomaly", (20, y), 0.50, 1)
    y += 26
    put_text(panel, "        -> Combined risk", (20, y), 0.50, 1)
    y += 26
    put_text(panel, "        -> CRI", (20, y), 0.50, 1)

    y += 45
    put_text(panel, "UCSD = abnormal-motion", (20, y), 0.48, 1)
    y += 23
    put_text(panel, "validation/demo", (20, y), 0.48, 1)

    return np.hstack([frame, panel])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not TEST_SEQ.exists():
        raise FileNotFoundError(
            f"Test sequence not found:\n{TEST_SEQ}"
        )

    frames = get_frames(TEST_SEQ)

    if len(frames) < 2:
        raise RuntimeError("Test003 does not contain enough frames.")

    model = learn_normal_model()

    # Separate background subtractor for the test sequence.
    bg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False,
    )

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError("Could not read first test frame.")

    height, width = first.shape[:2]

    # Make the demo easy to see on screen.
    display_scale = 2.0
    out_w = int(width * display_scale)
    out_h = int(height * display_scale)

    panel_w = min(430, max(360, out_w // 3))
    video_size = (out_w + panel_w, out_h)

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.5,
        video_size,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {OUTPUT_VIDEO}")

    prev_gray = None
    cri_history = deque(maxlen=CRI_WINDOW)

    # These are the previously frozen normal-validation thresholds.
    # They are shown for reference only; the demo also reports the
    # continuous CRI value so the calculation itself remains visible.
    medium_threshold = 0.82791673
    high_threshold = 0.87626211

    print("\nCreating UCSD Test003 CrowdSense MP4...")
    print(f"Output: {OUTPUT_VIDEO}")

    for idx, frame_path in enumerate(frames, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg = bg.apply(frame)
        density = float(np.mean(fg > 0))

        if prev_gray is None:
            mii = 0.0
            dci = 0.0
        else:
            flow = optical_flow(prev_gray, gray)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            valid = mag > 0.05

            if np.count_nonzero(valid) >= 10:
                m = mag[valid]
                a = ang[valid]
                mii = compute_mii(m)
                dci = compute_dci(a, m)
            else:
                mii = 0.0
                dci = 0.0

        density_n = normalize(density, DENSITY_MIN, DENSITY_MAX)
        mii_n = normalize(mii, MII_MIN, MII_MAX)

        cpi = compute_cpi(density_n, mii_n, dci)
        anomaly = normal_anomaly_score(model, density, mii, dci)

        combined = CPI_WEIGHT * cpi + ANOMALY_WEIGHT * anomaly

        cri_history.append(combined)
        cri = float(np.mean(cri_history))

        if cri >= high_threshold:
            risk = "HIGH"
        elif cri >= medium_threshold:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        values = {
            "density": density,
            "mii": mii,
            "dci": dci,
            "cpi": cpi,
            "anomaly": anomaly,
            "combined": combined,
            "cri": cri,
            "risk": risk,
        }

        # Enlarge source frame.
        display_frame = cv2.resize(
            frame,
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        )

        annotated = draw_panel(
            display_frame,
            values,
            idx,
            len(frames),
            12.5,
        )

        writer.write(annotated)

        prev_gray = gray

        if idx % 25 == 0 or idx == len(frames):
            print(
                f"Frame {idx:3d}/{len(frames)} | "
                f"Density={density:.4f} | "
                f"MII={mii:.4f} | "
                f"DCI={dci:.4f} | "
                f"CPI={cpi:.4f} | "
                f"CRI={cri:.4f} | "
                f"Risk={risk}"
            )

    writer.release()

    print("\nDemo video created successfully.")
    print(f"File: {OUTPUT_VIDEO}")
    print("You can open the MP4 and use it as the UCSD pipeline demonstration.")


if __name__ == "__main__":
    main()
