from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# CrowdSense - Final Evaluation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = Path(
    r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2"
)
TRAIN_ROOT = DATASET_ROOT / "UCSDped1" / "Train"
TEST_ROOT = DATASET_ROOT / "UCSDped1" / "Test"

# The normal-model CSV may be in the project root or outputs folder.
INPUT_CANDIDATES = [
    PROJECT_ROOT / "normal_model_results.csv",
    PROJECT_ROOT / "outputs" / "normal_model_results.csv",
]

INPUT = next((p for p in INPUT_CANDIDATES if p.exists()), None)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Frozen calibration values learned from UCSD Ped1 TRAIN data.
# Do not change these during final test evaluation.
# ------------------------------------------------------------
DENSITY_MIN = 0.00326960
DENSITY_MAX = 0.06023295
MII_MIN = 0.12297923
MII_MAX = 0.46781751

CRI_WINDOW = 5

# Current proposed CrowdSense risk combination.
CPI_WEIGHT = 0.7
ANOMALY_WEIGHT = 0.3

# Test001 contains no abnormal GT annotation in this packaging.
# It is used only as a NORMAL operational threshold-calibration
# sequence and is excluded from the annotated final test metrics.
THRESHOLD_VALIDATION_SEQUENCE = "Test001"


def compute_density(foreground_mask):
    """Foreground occupancy proxy."""
    if foreground_mask is None or foreground_mask.size == 0:
        raise ValueError("Invalid foreground mask.")

    return float(
        np.count_nonzero(foreground_mask > 0)
        / foreground_mask.size
    )


def compute_mii(magnitude):
    """Motion Instability Index = standard deviation of flow speed."""
    if magnitude is None or magnitude.size == 0:
        raise ValueError("Invalid optical-flow magnitude.")

    return float(np.std(magnitude.astype(np.float32)))


def compute_dci(angle, magnitude):
    """
    Directional Conflict Index based on circular variance.

    Only motion vectors above the median magnitude are used so that
    near-zero/background flow does not dominate directional statistics.
    """
    if angle is None or magnitude is None:
        raise ValueError("Angle and magnitude are required.")

    if angle.size == 0 or magnitude.size == 0:
        raise ValueError("Empty optical-flow arrays.")

    magnitude_threshold = max(
        float(np.percentile(magnitude, 50)),
        1e-6,
    )

    valid = magnitude > magnitude_threshold

    if not np.any(valid):
        return 0.0

    theta = np.deg2rad(angle[valid])

    mean_cos = float(np.mean(np.cos(theta)))
    mean_sin = float(np.mean(np.sin(theta)))

    resultant_length = np.sqrt(
        mean_cos ** 2 + mean_sin ** 2
    )

    return float(
        np.clip(1.0 - resultant_length, 0.0, 1.0)
    )


def extract_features(sequence_dir):
    """
    Recompute Density, MII and DCI using exactly the same
    preprocessing/optical-flow settings as the CrowdSense pipeline.
    """
    sequence_dir = Path(sequence_dir)

    frame_files = sorted(sequence_dir.glob("*.tif"))

    if not frame_files:
        raise FileNotFoundError(
            f"No .tif frames found in: {sequence_dir}"
        )

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False,
    )

    kernel = np.ones((3, 3), dtype=np.uint8)

    previous_gray = None
    rows = []

    for frame_number, frame_file in enumerate(frame_files, start=1):

        frame = cv2.imread(str(frame_file))

        if frame is None:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        foreground_mask = subtractor.apply(frame)

        foreground_mask = cv2.morphologyEx(
            foreground_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        foreground_mask = cv2.morphologyEx(
            foreground_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        d = compute_density(foreground_mask)

        # First frame has no previous frame for optical flow.
        if previous_gray is None:
            previous_gray = gray

            rows.append([
                frame_number,
                d,
                0.0,
                0.0,
            ])

            continue

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            0.5,   # pyr_scale
            3,     # levels
            15,    # winsize
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0,     # flags
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
            angleInDegrees=True,
        )

        motion_instability = compute_mii(magnitude)
        directional_conflict = compute_dci(
            angle,
            magnitude,
        )

        rows.append([
            frame_number,
            d,
            motion_instability,
            directional_conflict,
        ])

        previous_gray = gray

    return pd.DataFrame(
        rows,
        columns=[
            "frame",
            "density",
            "mii",
            "dci",
        ],
    )


def learn_normal_statistics():
    """
    Learn robust normal-behaviour statistics from UCSD Ped1 TRAIN.

    The test annotations are never used here.
    """
    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(
            f"UCSD Ped1 Train folder not found:\n{TRAIN_ROOT}"
        )

    feature_parts = []

    train_sequences = sorted(
        p for p in TRAIN_ROOT.iterdir()
        if p.is_dir()
    )

    if not train_sequences:
        raise RuntimeError(
            f"No training sequences found in: {TRAIN_ROOT}"
        )

    print(
        f"Found {len(train_sequences)} UCSD Ped1 training sequences."
    )

    for sequence_dir in train_sequences:
        print(
            f"  Learning normal features from {sequence_dir.name}..."
        )

        features = extract_features(sequence_dir)
        feature_parts.append(features)

    train = pd.concat(
        feature_parts,
        ignore_index=True,
    )

    statistics = {}

    for feature in ["density", "mii", "dci"]:

        values = train[feature].astype(float)

        median = float(values.median())

        mad = float(
            np.median(
                np.abs(values - median)
            )
        )

        robust_scale = 1.4826 * mad

        # Safety fallback only if MAD is effectively zero.
        if robust_scale < 1e-6:
            robust_scale = float(values.std())

        statistics[feature] = (
            median,
            max(robust_scale, 1e-6),
        )

    return statistics


def add_crowdsense_scores(df, statistics):
    """Compute CPI, anomaly score, combined risk and CRI."""

    density_normalized = (
        (df["density"] - DENSITY_MIN)
        / (DENSITY_MAX - DENSITY_MIN)
    ).clip(0.0, 1.0)

    mii_normalized = (
        (df["mii"] - MII_MIN)
        / (MII_MAX - MII_MIN)
    ).clip(0.0, 1.0)

    dci_normalized = df["dci"].clip(0.0, 1.0)

    # Crowd Pressure Index.
    df["cpi"] = (
        density_normalized
        * (
            0.5 * mii_normalized
            + 0.5 * dci_normalized
        )
    )

    # Deviation from learned normal behaviour.
    anomaly_scores = []

    for _, row in df.iterrows():

        absolute_z_scores = []

        for feature in ["density", "mii", "dci"]:

            median, scale = statistics[feature]

            z = (
                abs(float(row[feature]) - median)
                / scale
            )

            absolute_z_scores.append(z)

        mean_absolute_z = float(
            np.mean(absolute_z_scores)
        )

        anomaly = (
            1.0
            - np.exp(-mean_absolute_z / 2.0)
        )

        anomaly_scores.append(anomaly)

    df["normal_anomaly"] = np.clip(
        anomaly_scores,
        0.0,
        1.0,
    )

    # Combined risk.
    df["combined_risk"] = np.clip(
        CPI_WEIGHT * df["cpi"]
        + ANOMALY_WEIGHT * df["normal_anomaly"],
        0.0,
        1.0,
    )

    # Crowd Risk Index: temporal persistence.
    df["cri"] = (
        df["combined_risk"]
        .rolling(
            CRI_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    return df


def calibrate_operational_thresholds(statistics):
    """
    Calibrate operational risk thresholds on Test001, which is
    normal-only in the supplied UCSD Ped1 packaging.

    IMPORTANT:
    Test001 is NOT included in final annotated test metrics.
    """
    validation_dir = (
        TEST_ROOT
        / THRESHOLD_VALIDATION_SEQUENCE
    )

    print(
        f"\nCalibrating operational thresholds using "
        f"{THRESHOLD_VALIDATION_SEQUENCE}..."
    )

    validation = extract_features(validation_dir)

    validation = add_crowdsense_scores(
        validation,
        statistics,
    )

    medium_threshold = float(
        validation["cri"].quantile(0.95)
    )

    high_threshold = float(
        validation["cri"].quantile(0.99)
    )

    return medium_threshold, high_threshold


def evaluate_final_test(
    df,
    medium_threshold,
    high_threshold,
):
    """Compute final frame-level and per-sequence metrics."""

    # Test001 was used only for operational threshold calibration.
    test = df[
        df["sequence"]
        != THRESHOLD_VALIDATION_SEQUENCE
    ].copy()

    y_true = test["ground_truth"].astype(int)

    # Threshold-independent ranking metrics.
    roc_auc = roc_auc_score(
        y_true,
        test["cpi_plus_anomaly"],
    )

    pr_auc = average_precision_score(
        y_true,
        test["cpi_plus_anomaly"],
    )

    # Operational HIGH-risk decision.
    test["predicted_high"] = (
        test["cri"] >= high_threshold
    ).astype(int)

    y_pred = test["predicted_high"]

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    false_alarm_rate = (
        fp / max(tn + fp, 1)
    )

    detection_rate = (
        tp / max(tp + fn, 1)
    )

    # --------------------------------------------------------
    # Per-sequence results.
    # --------------------------------------------------------
    per_sequence_rows = []

    for sequence, group in test.groupby("sequence"):

        gy = group["ground_truth"].astype(int)

        if gy.nunique() == 2:

            sequence_auc = roc_auc_score(
                gy,
                group["cpi_plus_anomaly"],
            )

            sequence_pr_auc = average_precision_score(
                gy,
                group["cpi_plus_anomaly"],
            )

        else:
            sequence_auc = np.nan
            sequence_pr_auc = np.nan

        sequence_precision = precision_score(
            gy,
            group["predicted_high"],
            zero_division=0,
        )

        sequence_recall = recall_score(
            gy,
            group["predicted_high"],
            zero_division=0,
        )

        sequence_f1 = f1_score(
            gy,
            group["predicted_high"],
            zero_division=0,
        )

        per_sequence_rows.append([
            sequence,
            len(group),
            int(gy.sum()),
            sequence_auc,
            sequence_pr_auc,
            sequence_precision,
            sequence_recall,
            sequence_f1,
        ])

    per_sequence = pd.DataFrame(
        per_sequence_rows,
        columns=[
            "sequence",
            "frames",
            "abnormal_frames",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
        ],
    )

    return (
        test,
        {
            "test_sequences": test["sequence"].nunique(),
            "test_frames": len(test),
            "normal_frames": int((y_true == 0).sum()),
            "abnormal_frames": int((y_true == 1).sum()),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "medium_threshold": medium_threshold,
            "high_threshold": high_threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_rate": false_alarm_rate,
            "detection_rate": detection_rate,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        per_sequence,
    )


def main():

    print("=" * 60)
    print("CROWD SENSE - FINAL EVALUATION")
    print("=" * 60)

    if INPUT is None:
        print("\nERROR: normal_model_results.csv was not found.")
        print("Checked:")
        for candidate in INPUT_CANDIDATES:
            print(f"  {candidate}")
        print(
            "\nCopy normal_model_results.csv into the project "
            "root or outputs folder and run again."
        )
        raise SystemExit(1)

    if not TEST_ROOT.exists():
        raise FileNotFoundError(
            f"UCSD Ped1 Test folder not found:\n{TEST_ROOT}"
        )

    print(f"\nInput results: {INPUT}")

    # --------------------------------------------------------
    # 1. Learn normal model from TRAIN only.
    # --------------------------------------------------------
    print("\n[1/4] Learning normal behaviour from UCSD Ped1 TRAIN...")

    statistics = learn_normal_statistics()

    print("\nLearned robust statistics:")
    for feature, (median, scale) in statistics.items():
        print(
            f"  {feature:8s} "
            f"median={median:.6f} "
            f"scale={scale:.6f}"
        )

    # --------------------------------------------------------
    # 2. Operational threshold calibration.
    # --------------------------------------------------------
    print("\n[2/4] Calibrating operational thresholds...")

    medium_threshold, high_threshold = (
        calibrate_operational_thresholds(
            statistics
        )
    )

    print(
        f"  MEDIUM threshold (95th percentile): "
        f"{medium_threshold:.6f}"
    )

    print(
        f"  HIGH threshold   (99th percentile): "
        f"{high_threshold:.6f}"
    )

    threshold_file = (
        OUTPUT_DIR / "risk_thresholds.txt"
    )

    threshold_file.write_text(
        "\n".join([
            "CrowdSense operational thresholds",
            f"validation_sequence="
            f"{THRESHOLD_VALIDATION_SEQUENCE}",
            f"medium_threshold={medium_threshold:.8f}",
            f"high_threshold={high_threshold:.8f}",
            "",
            "Thresholds are calibrated on normal-only "
            "validation data and then frozen.",
        ]),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 3. Load final annotated test results.
    # --------------------------------------------------------
    print("\n[3/4] Evaluating annotated UCSD Ped1 TEST sequences...")

    results = pd.read_csv(INPUT)

    required_columns = {
        "sequence",
        "ground_truth",
        "cpi_plus_anomaly",
        "cri",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            "normal_model_results.csv is missing required "
            f"columns: {sorted(missing)}"
        )

    final_test, summary, per_sequence = (
        evaluate_final_test(
            results,
            medium_threshold,
            high_threshold,
        )
    )

    # --------------------------------------------------------
    # 4. Save all final outputs.
    # --------------------------------------------------------
    print("\n[4/4] Saving final evaluation outputs...")

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
        OUTPUT_DIR / "final_evaluation_summary.csv",
        index=False,
    )

    per_sequence.to_csv(
        OUTPUT_DIR / "final_per_sequence_results.csv",
        index=False,
    )

    final_test.to_csv(
        OUTPUT_DIR / "final_frame_level_results.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("FINAL CROWD SENSE RESULTS")
    print("=" * 60)

    print(
        f"ROC-AUC          : {summary['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC           : {summary['pr_auc']:.4f}"
    )

    print(
        f"MEDIUM threshold : {summary['medium_threshold']:.4f}"
    )

    print(
        f"HIGH threshold   : {summary['high_threshold']:.4f}"
    )

    print(
        f"Precision        : {summary['precision']:.4f}"
    )

    print(
        f"Recall           : {summary['recall']:.4f}"
    )

    print(
        f"F1               : {summary['f1']:.4f}"
    )

    print(
        f"False alarm rate : {summary['false_alarm_rate']:.4f}"
    )

    print(
        f"Detection rate   : {summary['detection_rate']:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        "                Pred Normal  Pred Abnormal"
    )
    print(
        f"Actual Normal       {summary['tn']:5d}"
        f"        {summary['fp']:5d}"
    )
    print(
        f"Actual Abnormal     {summary['fn']:5d}"
        f"        {summary['tp']:5d}"
    )

    print("\nOutput files:")
    print(
        f"  {OUTPUT_DIR / 'risk_thresholds.txt'}"
    )
    print(
        f"  {OUTPUT_DIR / 'final_evaluation_summary.csv'}"
    )
    print(
        f"  {OUTPUT_DIR / 'final_per_sequence_results.csv'}"
    )
    print(
        f"  {OUTPUT_DIR / 'final_frame_level_results.csv'}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
