import cv2
import os

# Path to one UCSD sequence
sequence_path = r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2\UCSDped1\Train\Train001"

image_files = sorted(os.listdir(sequence_path))

for image_name in image_files:

    image_path = os.path.join(sequence_path, image_name)

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Display all stages
    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blur)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()