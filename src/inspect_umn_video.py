import cv2
from pathlib import Path

video = Path(r"E:\PROJECTS\OPEN_CV\datasets\Crowd-Activity-All.avi")

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"Could not open: {video}")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frames / fps if fps else 0

print("=== Crowd-Activity-All.avi ===")
print(f"Path     : {video}")
print(f"FPS      : {fps:.3f}")
print(f"Frames   : {frames}")
print(f"Resolution: {width} x {height}")
print(f"Duration : {duration:.2f} seconds")

# Sample frames at 10%, 25%, 50%, 75%, 90%.
for pct in [0.10, 0.25, 0.50, 0.75, 0.90]:
    pos = int(frames * pct)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ok, frame = cap.read()
    if ok:
        out = Path("outputs") / f"umn_sample_{int(pct*100):02d}pct.jpg"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), frame)
        print(f"Saved sample {int(pct*100):02d}% -> {out}")

cap.release()
print("\nNext: use the metadata and samples to determine whether the file is one continuous video or contains multiple concatenated sequences.")
