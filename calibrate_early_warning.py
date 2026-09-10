from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math
import json

# ============================================================
# CrowdSense - Leakage-Free Threshold Calibration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(
    r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2"
)
TRAIN_ROOT = DATASET_ROOT / "UCSDped1" / "Train"
TEST_ROOT = DATASET_ROOT / "UCSDped1" / "Test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen feature calibration from UCSD Ped1 TRAIN.
DENSITY_MIN = 0.00326960
DENSITY_MAX = 0.06023295
MII_MIN = 0.12297923
MII_MAX = 0.46781751

CRI_WINDOW = 5
CPI_WEIGHT = 0.7
ANOMALY_WEIGHT = 0.3

FARNEBACK = (0.5, 3, 15, 3, 5, 1.2, 0)


def density(mask):
    return float(np.count_nonzero(mask > 0) / mask.size)


def mii(mag):
    return float(np.std(mag.astype(np.float32)))


def dci(angle, mag):
    threshold = max(float(np.percentile(mag, 50)), 1e-6)
    valid = mag > threshold
    if not np.any(valid):
        return 0.0
    theta = np.deg2rad(angle[valid])
    c = float(np.mean(np.cos(theta)))
    s = float(np.mean(np.sin(theta)))
    r = math.sqrt(c*c + s*s)
    return float(np.clip(1.0 - r, 0, 1))


def extract_features(sequence_dir):
    files = sorted(Path(sequence_dir).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No TIFF frames: {sequence_dir}")

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )
    kernel = np.ones((3, 3), np.uint8)

    prev = None
    rows = []

    for frame_no, f in enumerate(files, 1):
        frame = cv2.imread(str(f))
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        d = density(mask)

        if prev is None:
            rows.append([frame_no, d, 0.0, 0.0])
            prev = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None, *FARNEBACK
        )

        mag, angle = cv2.cartToPolar(
            flow[..., 0], flow[..., 1],
            angleInDegrees=True
        )

        rows.append([
            frame_no,
            d,
            mii(mag),
            dci(angle, mag)
        ])

        prev = gray

    return pd.DataFrame(
        rows,
        columns=["frame", "density", "mii", "dci"]
    )


def learn_normal_model():
    parts = []

    sequences = sorted(
        p for p in TRAIN_ROOT.iterdir()
        if p.is_dir()
    )

    if not sequences:
        raise RuntimeError("No UCSD Ped1 training sequences found.")

    print(f"Learning normal model from {len(sequences)} training sequences...")

    for seq in sequences:
        print(f"  {seq.name}")
        parts.append(extract_features(seq))

    train = pd.concat(parts, ignore_index=True)

    stats = {}

    for feature in ["density", "mii", "dci"]:
        values = train[feature].astype(float)
        med = float(values.median())
        mad = float(np.median(np.abs(values - med)))
        scale = max(1.4826 * mad, 1e-6)
        stats[feature] = {"median": med, "scale": scale}

    return stats


def add_scores(df, stats):
    dn = (
        (df["density"] - DENSITY_MIN)
        / (DENSITY_MAX - DENSITY_MIN)
    ).clip(0, 1)

    mn = (
        (df["mii"] - MII_MIN)
        / (MII_MAX - MII_MIN)
    ).clip(0, 1)

    dc = df["dci"].clip(0, 1)

    df["cpi"] = dn * (0.5 * mn + 0.5 * dc)

    anomalies = []

    for _, row in df.iterrows():
        zs = []

        for feature in ["density", "mii", "dci"]:
            med = stats[feature]["median"]
            scale = stats[feature]["scale"]
            zs.append(
                abs(float(row[feature]) - med) / scale
            )

        mean_z = float(np.mean(zs))
        anomaly = 1.0 - np.exp(-mean_z / 2.0)
        anomalies.append(anomaly)

    df["normal_anomaly"] = np.clip(anomalies, 0, 1)

    df["combined_risk"] = np.clip(
        CPI_WEIGHT * df["cpi"]
        + ANOMALY_WEIGHT * df["normal_anomaly"],
        0, 1
    )

    df["cri"] = (
        df["combined_risk"]
        .rolling(CRI_WINDOW, min_periods=1)
        .mean()
    )

    return df


def main():
    print("=" * 65)
    print("CROWDSENSE - LEAKAGE-FREE EARLY-WARNING CALIBRATION")
    print("=" * 65)

    stats = learn_normal_model()

    # --------------------------------------------------------
    # Build a NORMAL calibration distribution from TRAIN only.
    # We use the upper 99.5th percentile of TRAIN CRI.
    # This avoids using Test003 abnormal labels to select a threshold.
    # --------------------------------------------------------
    print("\nComputing CRI distribution on TRAIN normal behaviour...")

    train_cri_parts = []

    for seq in sorted(p for p in TRAIN_ROOT.iterdir() if p.is_dir()):
        df = extract_features(seq)
        df = add_scores(df, stats)
        train_cri_parts.append(df["cri"])

    train_cri = pd.concat(
        train_cri_parts,
        ignore_index=True
    )

    medium_threshold = float(train_cri.quantile(0.99))
    high_threshold = float(train_cri.quantile(0.995))

    # Ensure the operational labels remain ordered.
    high_threshold = max(high_threshold, medium_threshold)

    print("\nTRAIN-only operational thresholds:")
    print(f"MEDIUM = {medium_threshold:.6f}")
    print(f"HIGH   = {high_threshold:.6f}")

    threshold_data = {
        "method": "TRAIN-only CRI quantiles",
        "medium_quantile": 0.99,
        "high_quantile": 0.995,
        "medium_threshold": medium_threshold,
        "high_threshold": high_threshold,
        "feature_calibration": {
            "density_min": DENSITY_MIN,
            "density_max": DENSITY_MAX,
            "mii_min": MII_MIN,
            "mii_max": MII_MAX
        }
    }

    with open(
        OUTPUT_DIR / "crowdsense_frozen_thresholds.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(threshold_data, f, indent=2)

    # --------------------------------------------------------
    # Evaluate Test003 only AFTER threshold is frozen.
    # --------------------------------------------------------
    test_sequence = "Test003"

    print(f"\nEvaluating {test_sequence} with frozen thresholds...")

    test = extract_features(
        TEST_ROOT / test_sequence
    )
    test = add_scores(test, stats)

    # UCSD Test003 ground truth: abnormal begins at frame 91.
    # This is used ONLY after threshold selection.
    test["ground_truth"] = (
        test["frame"] >= 91
    ).astype(int)

    test["risk_level"] = np.where(
        test["cri"] >= high_threshold,
        "HIGH",
        np.where(
            test["cri"] >= medium_threshold,
            "MEDIUM",
            "LOW"
        )
    )

    # First sustained warning:
    # require 3 consecutive frames at/above HIGH threshold.
    high_flags = (
        test["cri"] >= high_threshold
    ).astype(int)

    sustained = (
        high_flags
        .rolling(3)
        .sum()
        .fillna(0)
        == 3
    )

    warning_frames = test.loc[
        sustained,
        "frame"
    ].tolist()

    first_warning = (
        int(warning_frames[0])
        if warning_frames
        else None
    )

    gt_start = 91

    if first_warning is not None:
        lead_time = gt_start - first_warning
    else:
        lead_time = None

    # False alarms before the ground-truth onset.
    pre_gt = test[test["frame"] < gt_start]

    false_alarm_frames = int(
        np.count_nonzero(
            pre_gt["cri"] >= high_threshold
        )
    )

    # Detection after ground-truth onset.
    post_gt = test[test["frame"] >= gt_start]

    detected_frames = int(
        np.count_nonzero(
            post_gt["cri"] >= high_threshold
        )
    )

    detection_rate = (
        detected_frames / len(post_gt)
        if len(post_gt)
        else 0.0
    )

    # Save frame-level analysis.
    test.to_csv(
        OUTPUT_DIR / "test003_early_warning_analysis.csv",
        index=False
    )

    summary = pd.DataFrame([{
        "sequence": test_sequence,
        "ground_truth_start_frame": gt_start,
        "medium_threshold": medium_threshold,
        "high_threshold": high_threshold,
        "first_sustained_high_warning_frame": first_warning,
        "early_warning_lead_frames": lead_time,
        "false_alarm_frames_before_gt": false_alarm_frames,
        "post_gt_high_detection_rate": detection_rate,
        "max_cri": float(test["cri"].max()),
        "mean_cri_normal": float(
            test.loc[test["ground_truth"] == 0, "cri"].mean()
        ),
        "mean_cri_abnormal": float(
            test.loc[test["ground_truth"] == 1, "cri"].mean()
        )
    }])

    summary.to_csv(
        OUTPUT_DIR / "test003_early_warning_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("EARLY-WARNING RESULT")
    print("=" * 65)
    print(f"Ground-truth congestion start : frame {gt_start}")
    print(f"MEDIUM threshold              : {medium_threshold:.4f}")
    print(f"HIGH threshold                : {high_threshold:.4f}")
    print(
        "First sustained HIGH warning  : "
        + (
            f"frame {first_warning}"
            if first_warning is not None
            else "NONE"
        )
    )
    print(
        "Early-warning lead            : "
        + (
            f"{lead_time} frames"
            if lead_time is not None
            else "NONE"
        )
    )
    print(
        f"False-alarm frames before GT  : {false_alarm_frames}"
    )
    print(
        f"Post-GT HIGH detection rate   : {detection_rate:.4f}"
    )
    print(
        f"Maximum CRI                   : {test['cri'].max():.4f}"
    )
    print(
        f"Mean CRI before GT            : "
        f"{test.loc[test['ground_truth'] == 0, 'cri'].mean():.4f}"
    )
    print(
        f"Mean CRI after GT             : "
        f"{test.loc[test['ground_truth'] == 1, 'cri'].mean():.4f}"
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "crowdsense_frozen_thresholds.json")
    print(OUTPUT_DIR / "test003_early_warning_analysis.csv")
    print(OUTPUT_DIR / "test003_early_warning_summary.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
