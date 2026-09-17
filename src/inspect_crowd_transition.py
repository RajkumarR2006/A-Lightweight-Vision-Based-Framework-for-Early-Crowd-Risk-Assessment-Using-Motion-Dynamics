"""
Inspect the first Crowd-Activity-All sequence around its normal-to-abnormal transition.

Run:
    python src/inspect_crowd_transition.py

Creates:
    outputs/crowd_transition_400_520.jpg

This is a visual verification step before using an onset frame for
early-risk evaluation.
"""

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "datasets" / "Crowd-Activity-All.avi"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO}")

frames_to_sample = list(range(400, 521, 10))
thumbs = []

for f in frames_to_sample:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ok, frame = cap.read()
    if not ok:
        continue

    thumb = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

    cv2.putText(
        thumb,
        f"Frame {f} | {f/30:.1f}s",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    thumbs.append(thumb)

cap.release()

cols = 4
rows = int(np.ceil(len(thumbs) / cols))
sheet = np.zeros((rows * 240, cols * 320, 3), dtype=np.uint8)

for i, thumb in enumerate(thumbs):
    r, c = divmod(i, cols)
    sheet[
        r*240:(r+1)*240,
        c*320:(c+1)*320
    ] = thumb

path = OUT / "crowd_transition_400_520.jpg"
cv2.imwrite(str(path), sheet)

print(f"Saved transition contact sheet: {path}")
print("Frames sampled: 400, 410, ..., 520")
