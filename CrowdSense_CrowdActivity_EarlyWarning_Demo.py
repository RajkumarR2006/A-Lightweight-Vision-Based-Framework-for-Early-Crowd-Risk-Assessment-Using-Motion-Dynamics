"""
CrowdSense — Final Crowd Activity All Early-Warning Demo
=========================================================

Creates ONE presentation-ready video from the already-computed CrowdSense
frame results. It does NOT recompute optical flow or run another experiment.

Input:
  E:\PROJECTS\OPEN_CV\datasets\Crowd-Activity-All.avi
  E:\PROJECTS\OPEN_CV\outputs\CrowdSense_Final_CrowdActivity_FrameResults.csv

Demo:
  First crowd-event transition, frames 400–580.
  Reference transition: frame 526 (diagnostic reference from the inspected clip).
  Early-warning threshold: CRI >= 0.645131, calibrated from the normal-only
  training distribution used in the previous early-warning analysis.

The video deliberately distinguishes:
  - EARLY WARNING: CrowdSense risk crossed the normal-derived CRI threshold.
  - REFERENCE EVENT: the known/inspected event transition is reached.

This is a demonstration of emerging crowd-risk detection, not proof of
multi-second congestion prediction across all scenarios.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------- Paths --------------------
ROOT = Path(r"E:\PROJECTS\OPEN_CV")
VIDEO_PATH = ROOT / "datasets" / "Crowd-Activity-All.avi"
CSV_PATH = ROOT / "outputs" / "CrowdSense_Final_CrowdActivity_FrameResults.csv"
OUT_DIR = ROOT / "outputs"
OUT_VIDEO = OUT_DIR / "CrowdSense_CrowdActivity_EarlyWarning_Demo.mp4"

# -------------------- Demo settings --------------------
START_FRAME = 400
END_FRAME = 580
REFERENCE_EVENT_FRAME = 526
CRI_THRESHOLD = 0.645131
FPS_FALLBACK = 30.0

def put_text(img, text, org, scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)

def panel(img, x1, y1, x2, y2):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

def bar(img, x, y, w, h, value, label):
    value = float(np.clip(value, 0, 1))
    cv2.rectangle(img, (x, y), (x+w, y+h), (70, 70, 70), 1)
    cv2.rectangle(img, (x, y), (x+int(w*value), y+h), (80, 210, 255), -1)
    put_text(img, f"{label}: {value:.3f}", (x, y-7), 0.48, 1)

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df.set_index("frame")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Could not open Crowd-Activity-All.avi")

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Enlarge the small 320x240 source for presentation.
    scale = 2.5
    out_w = int(width * scale)
    out_h = int(height * scale)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create output video.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME - 1)

    history = []
    warning_frame = None
    frames_written = 0

    print("=" * 64)
    print("CrowdSense — Early-Warning Demonstration")
    print("=" * 64)
    print(f"Input : {VIDEO_PATH}")
    print(f"Output: {OUT_VIDEO}")
    print(f"Demo frames: {START_FRAME}–{END_FRAME}")
    print(f"Reference event frame: {REFERENCE_EVENT_FRAME}")
    print(f"CRI warning threshold: {CRI_THRESHOLD:.6f}")
    print()

    for frame_no in range(START_FRAME, END_FRAME + 1):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_no not in df.index:
            continue

        row = df.loc[frame_no]
        cri = float(row["cri"])
        density = float(row["density"])
        mii = float(row["mii"])
        dci = float(row["dci"])
        cpi = float(row["cpi"])
        anomaly = float(row["normal_anomaly"])
        combined = float(row["combined_risk"])
        time_sec = float(row["time_sec"])

        history.append((frame_no, cri))
        if len(history) > 90:
            history.pop(0)

        # Online decision: only current/past CRI is used.
        warning = cri >= CRI_THRESHOLD
        if warning and warning_frame is None:
            warning_frame = frame_no

        is_reference_event = frame_no >= REFERENCE_EVENT_FRAME

        # Resize source.
        vis = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        # Header.
        cv2.rectangle(vis, (0, 0), (out_w, 62), (15, 15, 15), -1)
        put_text(vis, "CROWD SENSE", (20, 39), 1.0, 2)
        put_text(vis, "Temporal Crowd-Risk Assessment", (260, 38), 0.62, 2)

        # Right information panel.
        px1 = out_w - 360
        px2 = out_w - 15
        panel(vis, px1, 80, px2, out_h - 15)

        put_text(vis, "CROWDSENSE OUTPUT", (px1+18, 112), 0.62, 2)

        put_text(vis, f"Time      {time_sec:6.2f} s", (px1+18, 148), 0.52, 1)
        put_text(vis, f"Frame     {frame_no:6d}", (px1+18, 174), 0.52, 1)

        bar(vis, px1+18, 215, 300, 15, density, "Density")
        bar(vis, px1+18, 258, 300, 15, mii, "MII")
        bar(vis, px1+18, 301, 300, 15, dci, "DCI")
        bar(vis, px1+18, 344, 300, 15, cpi, "CPI")
        bar(vis, px1+18, 387, 300, 15, anomaly, "Anomaly")
        bar(vis, px1+18, 430, 300, 15, combined, "Combined Risk")
        bar(vis, px1+18, 473, 300, 15, cri, "CRI")

        # Prediction state.
        if is_reference_event:
            state = "EVENT IN PROGRESS"
        elif warning:
            state = "EARLY WARNING ACTIVE"
        else:
            state = "MONITORING"

        put_text(vis, "STATE", (px1+18, 520), 0.48, 1)
        put_text(vis, state, (px1+18, 550), 0.60, 2)

        # Prediction message.
        if warning and not is_reference_event:
            cv2.rectangle(vis, (px1+12, 570), (px2-12, 632), (0, 95, 180), -1)
            put_text(vis, "CROWD RISK PREDICTED", (px1+28, 607), 0.60, 2)
        elif is_reference_event:
            cv2.rectangle(vis, (px1+12, 570), (px2-12, 632), (0, 110, 170), -1)
            put_text(vis, "ABNORMAL EVENT ACTIVE", (px1+25, 607), 0.56, 2)
        else:
            cv2.rectangle(vis, (px1+12, 570), (px2-12, 632), (55, 55, 55), -1)
            put_text(vis, "NO EARLY WARNING", (px1+43, 607), 0.56, 2)

        # Live CRI trajectory at bottom-left.
        gx1, gy1 = 20, out_h - 185
        gx2, gy2 = px1 - 25, out_h - 25
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (20, 20, 20), -1)
        put_text(vis, "CRI temporal trajectory", (gx1+12, gy1+25), 0.50, 1)

        # Threshold line.
        ty = gy2 - int(CRI_THRESHOLD * (gy2-gy1-45))
        cv2.line(vis, (gx1+12, ty), (gx2-12, ty), (0, 180, 255), 1)
        put_text(vis, f"threshold {CRI_THRESHOLD:.2f}", (gx1+15, ty-5), 0.40, 1)

        if len(history) >= 2:
            vals = np.array([v for _, v in history], dtype=float)
            lo, hi = 0, 1
            plot_x1, plot_x2 = gx1+12, gx2-12
            plot_y1, plot_y2 = gy1+35, gy2-12
            pts = []
            for i, v in enumerate(vals):
                x = plot_x1 + int(i * (plot_x2-plot_x1) / max(1, len(vals)-1))
                y = plot_y2 - int(np.clip(v,0,1) * (plot_y2-plot_y1))
                pts.append((x, y))
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False,
                          (255, 255, 255), 2, cv2.LINE_AA)

        # Reference event marker.
        if frame_no >= REFERENCE_EVENT_FRAME:
            put_text(vis, "REFERENCE EVENT", (20, 92), 0.58, 2)
            cv2.rectangle(vis, (0, 100), (out_w-380, 130), (0, 90, 170), -1)
            put_text(vis, "Reference crowd-event transition reached",
                     (15, 122), 0.48, 1)

        # Pre-event warning marker.
        if warning_frame is not None and frame_no >= warning_frame and frame_no < REFERENCE_EVENT_FRAME:
            cv2.rectangle(vis, (15, 145), (out_w-380, 190), (0, 80, 170), -1)
            put_text(vis, "EARLY WARNING: CRI crossed normal-derived threshold",
                     (25, 176), 0.48, 1)

        writer.write(vis)
        frames_written += 1

    cap.release()
    writer.release()

    print(f"Frames written : {frames_written}")
    if warning_frame is not None:
        lead_frames = REFERENCE_EVENT_FRAME - warning_frame
        lead_seconds = lead_frames / fps
        print(f"Warning frame  : {warning_frame}")
        print(f"Reference frame: {REFERENCE_EVENT_FRAME}")
        print(f"Lead time      : {lead_seconds:.2f} seconds")
    else:
        print("Warning did not trigger before the reference event.")

    print(f"\nVIDEO CREATED:\n{OUT_VIDEO}")

if __name__ == "__main__":
    main()
