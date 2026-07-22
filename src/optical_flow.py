import cv2
import numpy as np
import os

# Dataset path
sequence_path = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Train\Train001"

image_files = sorted(os.listdir(sequence_path))

# Read the first frame
first_frame = cv2.imread(os.path.join(sequence_path, image_files[0]))

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

for image_name in image_files[1:]:

    frame = cv2.imread(os.path.join(sequence_path, image_name))

    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dense Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    # Convert vectors to magnitude and angle
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # Create HSV image for visualization
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    optical_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Original", frame)
    cv2.imshow("Optical Flow", optical_flow)

    prev_gray = gray

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()