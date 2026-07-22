import cv2
import os

# -----------------------------
# CHANGE THIS PATH
# -----------------------------
sequence_path = r"datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001"

# Get all image names
image_files = sorted(os.listdir(sequence_path))

print(f"Total Frames : {len(image_files)}")

# Read images one by one
for image_name in image_files:

    image_path = os.path.join(sequence_path, image_name)

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    cv2.imshow("UCSD Sequence", frame)

    # 30 ms delay (~33 FPS)
    key = cv2.waitKey(30)

    # Press 'q' to quit
    if key == ord('q'):
        break

cv2.destroyAllWindows()