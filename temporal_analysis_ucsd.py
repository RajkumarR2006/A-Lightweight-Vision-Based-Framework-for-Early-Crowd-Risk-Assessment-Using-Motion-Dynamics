import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Allow imports from src/
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INPUT = ROOT / "ucsd_calibrated_validation.csv"
OUTPUT = ROOT / "temporal_analysis_results.csv"

# Prototype alert threshold.
# IMPORTANT: this is exploratory only; do not report it as a final threshold.
THRESHOLD = 0.30
WINDOW = 5

df = pd.read_csv(INPUT)

# Actual validation CSV uses "ground_truth" for the frame-level label.
GT_COL = "ground_truth"
required = {"sequence", "frame", GT_COL, "cpi", "cri"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

rows = []

for seq, g in df.groupby("sequence"):
    g = g.sort_values("frame").reset_index(drop=True)

    gt = g[GT_COL].astype(int).to_numpy()
    cri = g["cri"].astype(float).to_numpy()
    cpi = g["cpi"].astype(float).to_numpy()

    abnormal_idx = np.where(gt == 1)[0]
    normal_idx = np.where(gt == 0)[0]

    if len(abnormal_idx) == 0:
        gt_start = None
        gt_end = None
    else:
        gt_start = int(g.loc[abnormal_idx[0], "frame"])
        gt_end = int(g.loc[abnormal_idx[-1], "frame"])

    # First CRI threshold crossing.
    crossings = np.where(cri >= THRESHOLD)[0]
    first_cross = int(g.loc[crossings[0], "frame"]) if len(crossings) else None

    if gt_start is not None and first_cross is not None:
        early_warning = gt_start - first_cross
    else:
        early_warning = None

    # False alarms before abnormality.
    if gt_start is None:
        pre_gt = g
    else:
        pre_gt = g[g["frame"] < gt_start]

    false_alarm_frames = int((pre_gt["cri"] >= THRESHOLD).sum())

    # Detection after GT starts.
    if gt_start is None:
        post_gt = g.iloc[0:0]
    else:
        post_gt = g[g["frame"] >= gt_start]

    detection_rate = (
        float((post_gt["cri"] >= THRESHOLD).mean())
        if len(post_gt) else np.nan
    )

    # Persistence: longest consecutive CRI alert run.
    alert = (cri >= THRESHOLD).astype(int)
    longest_run = 0
    current = 0
    for x in alert:
        if x:
            current += 1
            longest_run = max(longest_run, current)
        else:
            current = 0

    # Compare normal vs abnormal feature behavior.
    normal_mean_cri = (
        float(g.loc[gt == 0, "cri"].mean()) if len(normal_idx) else np.nan
    )
    abnormal_mean_cri = (
        float(g.loc[gt == 1, "cri"].mean()) if len(abnormal_idx) else np.nan
    )
    normal_mean_cpi = (
        float(g.loc[gt == 0, "cpi"].mean()) if len(normal_idx) else np.nan
    )
    abnormal_mean_cpi = (
        float(g.loc[gt == 1, "cpi"].mean()) if len(abnormal_idx) else np.nan
    )

    rows.append({
        "sequence": seq,
        "frames": len(g),
        "normal_frames": len(normal_idx),
        "abnormal_frames": len(abnormal_idx),
        "gt_start": gt_start,
        "gt_end": gt_end,
        "first_cri_crossing": first_cross,
        "early_warning_frames": early_warning,
        "false_alarm_frames_before_gt": false_alarm_frames,
        "post_gt_detection_rate": detection_rate,
        "longest_alert_run": longest_run,
        "normal_mean_cpi": normal_mean_cpi,
        "abnormal_mean_cpi": abnormal_mean_cpi,
        "normal_mean_cri": normal_mean_cri,
        "abnormal_mean_cri": abnormal_mean_cri,
    })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT, index=False)

print("\n=== TEMPORAL ANALYSIS ===")
print(f"Threshold: CRI >= {THRESHOLD}")
print(f"Input: {INPUT}")
print(f"Output: {OUTPUT}\n")

print(result.to_string(index=False))

print("\n=== TEST023 DETAIL ===")
t23 = result[result["sequence"].astype(str).str.contains("Test023", case=False)]
if len(t23):
    print(t23.to_string(index=False))
else:
    print("Test023 not found.")

print("\nDone.")
