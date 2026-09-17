"""
CrowdSense - Crowd Activity All early-risk test (first outdoor sequence)

Purpose:
    Use the FROZEN UCSD Ped1 normal model/calibration to process the first
    outdoor sequence in Crowd-Activity-All.avi and measure whether CRI
    starts rising before the visually verified transition around frame 480.

Important:
    - No retraining on this video.
    - No threshold tuning on the test sequence.
    - The 480 frame is treated as the reference transition for this diagnostic.
      The contact sheet should be interpreted as a transition interval:
      frames ~480-500 show the onset/development of the escape behaviour.

Run from project root:
    python src/run_crowd_activity_early_test.py

Outputs:
    outputs/crowd_activity_clip1_early_test.csv
    outputs/crowd_activity_clip1_early_risk.png
"""

from pathlib import Path
from collections import deque
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "datasets" / "Crowd-Activity-All.avi"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# First sequence: use a conservative range around the transition.
START_FRAME = 1
END_FRAME = 590

# Reference transition identified from the video inspection.
TRANSITION_FRAME = 480

# Frozen UCSD Ped1 feature calibration.
DENSITY_MIN = 0.00326960
DENSITY_MAX = 0.06023295
MII_MIN = 0.12297923
MII_MAX = 0.46781751

# Frozen UCSD normal-model statistics.
# These are the values learned from the 34 UCSD Ped1 training sequences.
DENSITY_MED = 0.024293
DENSITY_SCALE = 0.017959
MII_MED = 0.276059
MII_SCALE = 0.117566
DCI_MED = 0.834772
DCI_SCALE = 0.106145

CRI_WINDOW = 5


def normalize(x, lo, hi):
    return float(np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def compute_flow_features(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    valid = mag > 0.05

    if np.count_nonzero(valid) < 10:
        return 0.0, 0.0

    m = mag[valid].astype(np.float32)
    a = ang[valid].astype(np.float32)

    # MII = standard deviation of flow magnitude.
    mii = float(np.std(m))

    # DCI = circular variance of strong motion directions.
    threshold = max(float(np.median(m)), 1e-6)
    strong = m > threshold

    if np.count_nonzero(strong) < 10:
        strong = m > 1e-6

    if np.count_nonzero(strong) == 0:
        return mii, 0.0

    aa = a[strong]
    mean_cos = float(np.mean(np.cos(aa)))
    mean_sin = float(np.mean(np.sin(aa)))
    r = np.sqrt(mean_cos**2 + mean_sin**2)
    dci = float(np.clip(1.0 - r, 0.0, 1.0))

    return mii, dci


def normal_anomaly(density, mii, dci):
    zd = abs(density - DENSITY_MED) / max(DENSITY_SCALE, 1e-6)
    zm = abs(mii - MII_MED) / max(MII_SCALE, 1e-6)
    zc = abs(dci - DCI_MED) / max(DCI_SCALE, 1e-6)

    mean_abs_z = (zd + zm + zc) / 3.0
    return float(1.0 - np.exp(-mean_abs_z / 2.0))


def process():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME - 1)

    bg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )

    rows = []
    prev_gray = None
    cri_history = deque(maxlen=CRI_WINDOW)

    for frame_no in range(START_FRAME, END_FRAME + 1):
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Same foreground/background subtraction configuration used by
        # the CrowdSense pipeline.
        foreground = bg.apply(frame)
        density = float(np.mean(foreground > 0))

        if prev_gray is None:
            mii, dci = 0.0, 0.0
        else:
            mii, dci = compute_flow_features(prev_gray, gray)

        density_n = normalize(density, DENSITY_MIN, DENSITY_MAX)
        mii_n = normalize(mii, MII_MIN, MII_MAX)

        # Current CrowdSense CPI definition.
        cpi = density_n * (0.5 * mii_n + 0.5 * dci)

        anomaly = normal_anomaly(density, mii, dci)

        # Current combined risk used by the integrated pipeline.
        combined = 0.7 * cpi + 0.3 * anomaly

        cri_history.append(combined)
        cri = float(np.mean(cri_history))

        rows.append({
            "frame": frame_no,
            "time_sec": (frame_no - 1) / fps,
            "density": density,
            "mii": mii,
            "dci": dci,
            "density_norm": density_n,
            "mii_norm": mii_n,
            "cpi": cpi,
            "normal_anomaly": anomaly,
            "combined_risk": combined,
            "cri": cri,
            "pre_transition": int(frame_no < TRANSITION_FRAME),
        })

        prev_gray = gray

    cap.release()
    return rows, fps


def save_csv(rows):
    path = OUT / "crowd_activity_clip1_early_test.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_plot(rows, fps):
    frames = np.array([r["frame"] for r in rows])
    cri = np.array([r["cri"] for r in rows])
    cpi = np.array([r["cpi"] for r in rows])
    anomaly = np.array([r["normal_anomaly"] for r in rows])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frames, cri, label="CRI", linewidth=2)
    ax.plot(frames, cpi, label="CPI", alpha=0.75)
    ax.plot(frames, anomaly, label="Normal anomaly", alpha=0.75)

    ax.axvline(
        TRANSITION_FRAME,
        linestyle="--",
        linewidth=2,
        label=f"Reference transition: frame {TRANSITION_FRAME}"
    )

    ax.axvspan(
        TRANSITION_FRAME,
        frames[-1],
        alpha=0.08,
        label="Post-transition"
    )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Risk score")
    ax.set_title("CrowdSense Early-Risk Test — Crowd Activity All")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = OUT / "crowd_activity_clip1_early_risk.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def report(rows, fps):
    print("\n=== Crowd Activity All: First Sequence ===")
    print(f"FPS                  : {fps:.2f}")
    print(f"Frames processed     : {rows[0]['frame']} - {rows[-1]['frame']}")
    print(f"Reference transition : frame {TRANSITION_FRAME}")
    print(f"Reference time       : {(TRANSITION_FRAME - 1)/fps:.2f} s")

    print("\nRisk immediately before transition:")
    for f in [390, 420, 450, 460, 470, 480, 490, 500, 510]:
        match = [r for r in rows if r["frame"] == f]
        if match:
            r = match[0]
            print(
                f"Frame {f:3d}: "
                f"Density={r['density']:.4f}  "
                f"MII={r['mii']:.4f}  "
                f"DCI={r['dci']:.4f}  "
                f"CPI={r['cpi']:.4f}  "
                f"CRI={r['cri']:.4f}"
            )

    print("\nMean CRI by pre-transition window:")
    for end in [420, 440, 460, 470, 479]:
        start = end - 29
        vals = [r["cri"] for r in rows if start <= r["frame"] <= end]
        if vals:
            print(
                f"Frames {start}-{end}: "
                f"mean CRI={np.mean(vals):.4f}, "
                f"max CRI={np.max(vals):.4f}"
            )

    pre = [r for r in rows if r["frame"] < TRANSITION_FRAME]
    post = [r for r in rows if r["frame"] >= TRANSITION_FRAME]

    max_pre = max(pre, key=lambda r: r["cri"])
    max_post = max(post, key=lambda r: r["cri"])

    print("\nSummary:")
    print(
        f"Peak pre-transition CRI  = {max_pre['cri']:.4f} "
        f"at frame {max_pre['frame']}"
    )
    print(
        f"Peak post-transition CRI = {max_post['cri']:.4f} "
        f"at frame {max_post['frame']}"
    )

    print("\nNo warning threshold is applied in this diagnostic.")
    print("We first inspect whether the continuous CRI rises before the transition.")
    print("This avoids threshold-hunting on the evaluation video.")


def main():
    rows, fps = process()

    if not rows:
        raise RuntimeError("No frames processed.")

    csv_path = save_csv(rows)
    plot_path = save_plot(rows, fps)

    report(rows, fps)

    print("\nSaved:")
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
