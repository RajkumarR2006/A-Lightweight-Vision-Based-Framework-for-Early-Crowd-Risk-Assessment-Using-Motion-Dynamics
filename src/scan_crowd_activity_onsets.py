"""
CrowdSense - Crowd Activity All: all-sequence onset scan

Purpose:
    Detect the red "Abnormal Crowd Activity" ground-truth marker embedded
    in Crowd-Activity-All.avi and identify the abnormal-onset frame for each
    of the 11 clips.

This is NOT model training and does not change CrowdSense.

Run from project root:
    python src/scan_crowd_activity_onsets.py

Output:
    outputs/crowd_activity_onsets.csv

After this scan, the same frozen CrowdSense pipeline will be evaluated
around every detected onset. We will not tune thresholds per sequence.
"""

from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "datasets" / "Crowd-Activity-All.avi"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)

# The UMN/Crowd Activity All video embeds a red "Abnormal Crowd Activity"
# text marker in the upper-left corner during the abnormal portion.
def red_marker_score(frame):
    h, w = frame.shape[:2]
    roi = frame[:max(55, int(h * 0.18)), :max(190, int(w * 0.62))]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Red hue wraps around HSV.
    m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)

    # Text-like red pixels. Use count and density, not a single pixel.
    count = int(np.count_nonzero(mask))
    density = count / mask.size
    return count, density

scores = []
frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_no += 1
    count, density = red_marker_score(frame)
    scores.append((frame_no, count, density))

cap.release()

counts = np.array([x[1] for x in scores], dtype=np.float32)

# Inspect the distribution rather than hard-coding a dataset-specific count.
# The marker is a large red text overlay, so its count should be an obvious
# high-density mode.
q90 = float(np.percentile(counts, 90))
q99 = float(np.percentile(counts, 99))
threshold = max(40.0, q99 * 0.25)

is_red = counts >= threshold

# Require persistence to suppress isolated red objects.
run = 0
onsets = []
for i, flag in enumerate(is_red):
    if flag:
        run += 1
    else:
        if run >= 5:
            start_index = i - run
            onsets.append(scores[start_index][0])
        run = 0

if run >= 5:
    onsets.append(scores[len(scores) - run][0])

# Deduplicate nearby detections.
clean = []
for f in onsets:
    if not clean or f - clean[-1] > int(fps * 5):
        clean.append(f)

# Save all frame scores for auditability.
score_path = OUT / "crowd_activity_red_marker_scores.csv"
with score_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["frame", "red_pixel_count", "red_density"])
    writer.writerows(scores)

onset_path = OUT / "crowd_activity_onsets.csv"
with onset_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["clip", "abnormal_onset_frame", "time_sec"])
    for clip, f in enumerate(clean, start=1):
        writer.writerow([clip, f, (f - 1) / fps])

print("=== Crowd Activity All: Embedded Ground-Truth Marker Scan ===")
print(f"FPS                 : {fps:.2f}")
print(f"Total frames        : {len(scores)}")
print(f"90th percentile red : {q90:.1f}")
print(f"99th percentile red : {q99:.1f}")
print(f"Marker threshold    : {threshold:.1f}")
print(f"Detected onsets     : {len(clean)}")

for i, f in enumerate(clean, start=1):
    print(f"Clip {i:2d}: frame {f:5d} | time {(f-1)/fps:7.2f}s")

print("\nSaved:")
print(score_path)
print(onset_path)
