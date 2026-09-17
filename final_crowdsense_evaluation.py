"""
CrowdSense FINAL evaluation
- Learns normal behaviour from UCSD Ped1 Train (34 sequences)
- Runs the frozen CrowdSense feature pipeline on Crowd-Activity-All.avi
- Uses the same frozen feature calibration and normal-model formulation
- Evaluates 11 reference crowd-event onsets
- Produces ONE CSV and ONE PNG; no intermediate scripts required.

Run from E:\PROJECTS\OPEN_CV:
    python final_crowdsense_evaluation.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = r"E:\PROJECTS\OPEN_CV"
VIDEO = os.path.join(ROOT, "datasets", "Crowd-Activity-All.avi")
TRAIN_ROOT = os.path.join(ROOT, "datasets", "UCSD_Anomaly_Dataset.v1p2", "UCSDped1", "Train")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Frozen CrowdSense feature calibration
DENSITY_MIN = 0.00326960
DENSITY_MAX = 0.06023295
MII_MIN = 0.12297923
MII_MAX = 0.46781751

# CrowdSense formulation
CRI_WINDOW = 5
CPI_WEIGHT = 0.7
ANOMALY_WEIGHT = 0.3

# Reference abnormal-event onset frames in the concatenated Crowd Activity All video.
# These are the 11 scenario-level candidates retained from the marker scan.
# They are treated as reference event onsets, NOT as perfect ground truth.
ONSETS = [485, 1074, 1331, 1807, 2606, 3220, 3939, 4808, 5423, 6196, 6884]

# Three published UMN scenario reference points are especially well documented:
# scenario 1 ~= 526, scenario 3 ~= 1770, scenario 9 ~= 6172.
# Our marker-derived values above are used consistently for all 11 scenarios.
FPS_EXPECTED = 30.0

def normalize(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def compute_density(mask):
    return float(np.count_nonzero(mask) / mask.size)

def compute_mii(magnitude):
    return float(np.std(magnitude.astype(np.float32)))

def compute_dci(angle, magnitude):
    # Ignore near-zero flow; use pixels above the median flow magnitude.
    valid = magnitude > np.median(magnitude)
    if np.count_nonzero(valid) < 10:
        valid = magnitude > 0.1
    if np.count_nonzero(valid) == 0:
        return 0.0
    theta = angle[valid]
    mean_cos = float(np.mean(np.cos(theta)))
    mean_sin = float(np.mean(np.sin(theta)))
    R = np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
    return float(np.clip(1.0 - R, 0.0, 1.0))

def process_frame(prev_gray, gray, bg):
    fg = bg.apply(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)

    # Remove tiny noise and shadows/artifacts.
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    density = compute_density(fg)
    mii = compute_mii(mag)
    dci = compute_dci(ang, mag)

    dn = normalize(density, DENSITY_MIN, DENSITY_MAX)
    mn = normalize(mii, MII_MIN, MII_MAX)
    cpi = dn * (0.5 * mn + 0.5 * dci)

    return density, mii, dci, cpi

def robust_stats(values):
    values = np.asarray(values, dtype=np.float32)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = max(1.4826 * mad, 1e-6)
    return med, scale

def learn_normal_model():
    density_vals, mii_vals, dci_vals = [], [], []
    seqs = sorted(
        d for d in os.listdir(TRAIN_ROOT)
        if d.lower().startswith("train") and os.path.isdir(os.path.join(TRAIN_ROOT, d))
    )

    print(f"Learning normal behaviour from {len(seqs)} UCSD Ped1 training sequences...")

    for idx, seq in enumerate(seqs, 1):
        folder = os.path.join(TRAIN_ROOT, seq)
        files = sorted(
            [f for f in os.listdir(folder) if f.lower().endswith((".tif", ".bmp", ".jpg", ".png"))]
        )
        if len(files) < 2:
            continue

        bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        prev = None

        for name in files:
            img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if prev is None:
                bg.apply(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                prev = img
                continue

            d, m, dc, _ = process_frame(prev, img, bg)
            density_vals.append(d)
            mii_vals.append(m)
            dci_vals.append(dc)
            prev = img

        if idx % 10 == 0 or idx == len(seqs):
            print(f"  trained {idx}/{len(seqs)} sequences")

    stats = {
        "density": robust_stats(density_vals),
        "mii": robust_stats(mii_vals),
        "dci": robust_stats(dci_vals),
    }
    print("Normal model learned.")
    for k, (med, scale) in stats.items():
        print(f"  {k:7s}: median={med:.6f}, scale={scale:.6f}")
    return stats

def normal_anomaly(density, mii, dci, stats):
    vals = []
    for x, key in [(density, "density"), (mii, "mii"), (dci, "dci")]:
        med, scale = stats[key]
        vals.append(abs(x - med) / scale)
    mean_abs_z = float(np.mean(vals))
    return float(1.0 - np.exp(-mean_abs_z / 2.0))

def run_video(stats):
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_EXPECTED
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {total} frames @ {fps:.2f} FPS")
    print("Running one-pass CrowdSense evaluation...")

    rows = []
    bg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    prev = None
    frame_no = 0
    cri_history = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is None:
            bg.apply(frame)
            prev = gray
            continue

        density, mii, dci, cpi = process_frame(prev, gray, bg)
        anomaly = normal_anomaly(density, mii, dci, stats)
        combined = CPI_WEIGHT * cpi + ANOMALY_WEIGHT * anomaly

        cri_history.append(combined)
        cri = float(np.mean(cri_history[-CRI_WINDOW:]))

        rows.append({
            "frame": frame_no,
            "time_sec": frame_no / fps,
            "density": density,
            "mii": mii,
            "dci": dci,
            "cpi": cpi,
            "normal_anomaly": anomaly,
            "combined_risk": combined,
            "cri": cri,
        })

        prev = gray

        if frame_no % 500 == 0 or frame_no == total:
            print(f"  processed {frame_no}/{total} frames")

    cap.release()
    return pd.DataFrame(rows), fps

def evaluate_onsets(df, fps):
    # Fixed windows. This is a temporal trajectory test, not a tuned classifier.
    # PRE: 5 to 2 seconds before reference onset
    # LATE_PRE: 2 to 0.25 seconds before onset
    # POST: 0 to 3 seconds after onset
    results = []

    for i, onset in enumerate(ONSETS, 1):
        pre = df[(df.frame >= onset - int(5*fps)) &
                 (df.frame <  onset - int(2*fps))]
        late = df[(df.frame >= onset - int(2*fps)) &
                  (df.frame <  onset - int(0.25*fps))]
        post = df[(df.frame >= onset) &
                  (df.frame <= onset + int(3*fps))]

        if len(pre) == 0 or len(late) == 0 or len(post) == 0:
            continue

        pre_med = float(pre.cri.median())
        late_med = float(late.cri.median())
        post_max = float(post.cri.max())

        # A descriptive temporal-rise measure. No model threshold is tuned here.
        rise = late_med - pre_med
        rise_pct = 100.0 * rise / max(abs(pre_med), 1e-6)

        results.append({
            "scenario": i,
            "reference_onset_frame": onset,
            "reference_onset_sec": onset / fps,
            "pre_5_to_2s_median_cri": pre_med,
            "pre_2_to_0.25s_median_cri": late_med,
            "post_0_to_3s_max_cri": post_max,
            "pre_event_rise": rise,
            "pre_event_rise_percent": rise_pct,
        })

    return pd.DataFrame(results)

def make_plot(df, eval_df, fps):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df.time_sec, df.cri, linewidth=1.2, label="CrowdSense CRI")

    for _, r in eval_df.iterrows():
        ax.axvline(r.reference_onset_sec, linestyle="--", linewidth=0.9)
        ax.text(
            r.reference_onset_sec, 0.98,
            f"S{int(r.scenario)}",
            transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="right", fontsize=8
        )

    ax.set_title("CrowdSense — Crowd Activity All: CRI Temporal Trajectory")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("CRI")
    ax.set_ylim(0, max(1.0, float(df.cri.max()) * 1.05))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "CrowdSense_Final_CrowdActivity_CRI.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def main():
    if not os.path.exists(VIDEO):
        raise FileNotFoundError(VIDEO)
    if not os.path.isdir(TRAIN_ROOT):
        raise FileNotFoundError(TRAIN_ROOT)

    print("=" * 72)
    print("CrowdSense FINAL EVALUATION")
    print("=" * 72)

    stats = learn_normal_model()
    df, fps = run_video(stats)

    raw_csv = os.path.join(OUT_DIR, "CrowdSense_Final_CrowdActivity_FrameResults.csv")
    df.to_csv(raw_csv, index=False)

    eval_df = evaluate_onsets(df, fps)
    eval_csv = os.path.join(OUT_DIR, "CrowdSense_Final_CrowdActivity_Evaluation.csv")
    eval_df.to_csv(eval_csv, index=False)

    plot_path = make_plot(df, eval_df, fps)

    print("\n" + "=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)
    print(eval_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    positive = int((eval_df.pre_event_rise > 0).sum())
    total = len(eval_df)
    print(f"\nPre-event CRI increased in {positive}/{total} scenarios.")
    print(f"Frame results : {raw_csv}")
    print(f"Evaluation    : {eval_csv}")
    print(f"Plot          : {plot_path}")

    if positive == total:
        print("\nConclusion: CRI shows a consistent pre-event rise across all reference scenarios.")
    else:
        print("\nConclusion: CRI does NOT show a consistent pre-event rise across all scenarios.")
        print("Therefore, the current evidence supports temporal abnormal-event detection,")
        print("but not a blanket claim of reliable early congestion prediction.")

if __name__ == "__main__":
    main()
