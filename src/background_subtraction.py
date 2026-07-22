import cv2
import os

# Dataset path
sequence_path = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Train\Train001"

# Read all images
image_files = sorted(os.listdir(sequence_path))

# Create Background Subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

for image_name in image_files:

    image_path = os.path.join(sequence_path, image_name)

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Background subtraction
    foreground = back_sub.apply(blur)

    # Display
    cv2.imshow("Original", frame)
    cv2.imshow("Foreground Mask", foreground)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()