


import cv2

# Replace with your image path
image_path = r"datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
else:
    print("Image loaded successfully!\n")

    # Image properties
    print("Image Shape :", image.shape)
    print("Image Height:", image.shape[0], "pixels")
    print("Image Width :", image.shape[1], "pixels")
    print("Channels    :", image.shape[2])
    print("Data Type   :", image.dtype)

    cv2.imshow("First Frame", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()