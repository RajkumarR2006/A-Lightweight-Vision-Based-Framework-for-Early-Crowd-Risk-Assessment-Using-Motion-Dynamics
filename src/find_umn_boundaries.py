"""
Find candidate sequence boundaries in the combined UMN Crowd-Activity-All AVI.

Run from the project root:
    python src/find_umn_boundaries.py

Output:
    outputs/umn_candidate_boundaries.jpg

This only proposes visual sequence boundaries. We will inspect the
candidates before selecting a sequence for CrowdSense evaluation.
"""

from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO = PROJECT_ROOT / "datasets" / "Crowd-Activity-All.avi"
OUT = PROJECT_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scores = []
prev = None
frame_no = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_no += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (80, 60), interpolation=cv2.INTER_AREA)

    if prev is not None:
        scores.append(
            (frame_no, float(np.mean(cv2.absdiff(gray, prev))))
        )

    prev = gray

cap.release()

if not scores:
    raise RuntimeError("No frame-to-frame comparisons were produced.")

arr = np.array([x[1] for x in scores], dtype=np.float32)

# Strong frame-to-frame jumps are candidate sequence cuts.
threshold = max(
    float(np.percentile(arr, 99.5)),
    float(np.median(arr) + 6 * np.std(arr))
)

candidates = []
min_gap = max(1, int(fps * 3))

for i in np.argsort(arr)[::-1]:
    f = scores[int(i)][0]

    if all(abs(f - old_f) > min_gap for old_f, _ in candidates):
        candidates.append((f, float(arr[int(i)])))

    if len(candidates) >= 20:
        break

candidates.sort()

thumbs = []
cap = cv2.VideoCapture(str(VIDEO))

for f, score in candidates:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f - 1))
    ok, frame = cap.read()

    if not ok:
        continue

    thumb = cv2.resize(frame, (240, 180))

    cv2.putText(
        thumb,
        f"frame {f} | {f/fps:.1f}s",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    thumbs.append(thumb)

cap.release()

cols = 4
rows = max(1, int(np.ceil(len(thumbs) / cols)))
sheet = np.zeros((rows * 180, cols * 240, 3), dtype=np.uint8)

for i, thumb in enumerate(thumbs):
    r, c = divmod(i, cols)
    sheet[
        r * 180:(r + 1) * 180,
        c * 240:(c + 1) * 240
    ] = thumb

sheet_path = OUT / "umn_candidate_boundaries.jpg"
cv2.imwrite(str(sheet_path), sheet)

print(f"FPS       : {fps:.2f}")
print(f"Frames    : {n}")
print(f"Duration  : {n/fps:.2f} seconds")
print(f"Threshold : {threshold:.3f}")

print("\nCandidate visual boundaries:")
for f, score in candidates:
    print(
        f"frame {f:5d} | "
        f"{f/fps:7.2f}s | "
        f"difference {score:.3f}"
    )

print("\nContact sheet saved to:")
print(sheet_path)

print("\nNEXT: inspect the contact sheet before selecting a sequence.")
