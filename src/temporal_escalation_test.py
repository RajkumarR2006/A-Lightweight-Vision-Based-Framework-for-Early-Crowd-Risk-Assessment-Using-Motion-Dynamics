"""
CrowdSense - Temporal Escalation Diagnostic

ONE focused experiment:
    Measure whether the existing CrowdSense features show a sustained
    upward trajectory BEFORE the observed crowd-event transition.

No threshold tuning.
No retraining.
No new risk formula.
No modification of main.py.

Input:
    outputs/crowd_activity_clip1_early_test.csv

Reference transition:
    frame 480

Outputs:
    outputs/crowd_activity_temporal_escalation.csv
    outputs/crowd_activity_temporal_escalation.png

Run from project root:
    python src/temporal_escalation_test.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "crowd_activity_clip1_early_test.csv"
OUT = ROOT / "outputs"

TRANSITION = 480
WINDOW = 30

if not INPUT.exists():
    raise FileNotFoundError(f"Missing input: {INPUT}")

df = pd.read_csv(INPUT)
df = df.sort_values("frame").reset_index(drop=True)

# Rolling linear-regression slope over the previous WINDOW frames.
# Multiplying by 30 converts it to change per 30 frames (~1 second).
def rolling_slope(series, window):
    x = np.arange(window, dtype=np.float64)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)

    values = series.to_numpy(dtype=np.float64)
    slopes = np.full(len(values), np.nan)

    for i in range(window - 1, len(values)):
        y = values[i - window + 1:i + 1]
        y_centered = y - y.mean()
        slopes[i] = np.sum(x_centered * y_centered) / denom

    return slopes * 30.0


features = {
    "cri": "CRI",
    "cpi": "CPI",
    "mii": "MII",
    "dci": "DCI",
    "density": "Density",
}

for col in features:
    df[f"{col}_slope_30f"] = rolling_slope(df[col], WINDOW)

# A purely descriptive temporal escalation indicator:
# count how many of CRI/MII/DCI/Density have positive 30-frame slopes.
# This is NOT used as a prediction threshold.
slope_cols = [
    "cri_slope_30f",
    "mii_slope_30f",
    "dci_slope_30f",
    "density_slope_30f",
]
df["positive_slope_count"] = (
    df[slope_cols].gt(0).sum(axis=1)
)

out_csv = OUT / "crowd_activity_temporal_escalation.csv"
df.to_csv(out_csv, index=False)

# -------------------------------------------------------------
# Print focused measurements around the transition.
# -------------------------------------------------------------
print("=== CrowdSense Temporal Escalation Diagnostic ===")
print(f"Reference transition : frame {TRANSITION}")
print(f"Rolling window       : {WINDOW} frames (~1 second)")
print("\n30-frame slopes (change over approximately 1 second):")

check_frames = [360, 390, 420, 440, 450, 460, 470, 479, 480, 490, 500]

for frame in check_frames:
    row = df.loc[df["frame"] == frame]
    if row.empty:
        continue

    r = row.iloc[0]
    print(
        f"Frame {frame:3d}: "
        f"CRI slope={r['cri_slope_30f']:+.4f} | "
        f"MII slope={r['mii_slope_30f']:+.4f} | "
        f"DCI slope={r['dci_slope_30f']:+.4f} | "
        f"Density slope={r['density_slope_30f']:+.4f} | "
        f"positive={int(r['positive_slope_count'])}/4"
    )

print("\nMean CRI slope by pre-transition windows:")
for start, end in [(360, 389), (390, 419), (420, 449), (450, 479)]:
    vals = df.loc[
        (df["frame"] >= start) & (df["frame"] <= end),
        "cri_slope_30f"
    ].dropna()

    if len(vals):
        print(
            f"Frames {start}-{end}: "
            f"mean={vals.mean():+.5f}, "
            f"median={vals.median():+.5f}"
        )

# -------------------------------------------------------------
# Plot all temporal slopes in a single readable figure.
# -------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

plot_items = [
    ("cri_slope_30f", "CRI slope / 30 frames"),
    ("mii_slope_30f", "MII slope / 30 frames"),
    ("dci_slope_30f", "DCI slope / 30 frames"),
    ("density_slope_30f", "Density slope / 30 frames"),
]

for ax, (col, label) in zip(axes, plot_items):
    ax.plot(df["frame"], df[col], linewidth=1.5)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(
        TRANSITION,
        linestyle="--",
        linewidth=2,
        label=f"Transition frame {TRANSITION}"
    )
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel("Frame")
fig.suptitle(
    "CrowdSense Temporal Escalation — Crowd Activity All",
    fontsize=16
)
fig.tight_layout()

out_plot = OUT / "crowd_activity_temporal_escalation.png"
fig.savefig(out_plot, dpi=160)
plt.close(fig)

print("\nSaved:")
print(out_csv)
print(out_plot)
print("\nInterpretation rule:")
print(
    "We are looking for a sustained positive trajectory BEFORE frame 480. "
    "No warning threshold is introduced in this experiment."
)
