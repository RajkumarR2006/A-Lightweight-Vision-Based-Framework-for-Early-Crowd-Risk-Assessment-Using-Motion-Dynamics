import numpy as np


def compute_dci(angle, magnitude, motion_threshold=0.5):
    """
    Compute Direction Conflict Index (DCI) using only meaningful motion pixels.

    Parameters:
        angle : Optical flow directions (degrees)
        magnitude : Optical flow magnitudes
        motion_threshold : Minimum magnitude to consider

    Returns:
        dci : Direction Conflict Index (0 to 1)
    """

    if angle.size == 0 or magnitude.size == 0:
        return 0.0

    # Keep only pixels with significant motion
    mask = magnitude > motion_threshold

    if np.count_nonzero(mask) == 0:
        return 0.0

    filtered_angles = angle[mask]

    # Convert degrees to radians
    theta = np.deg2rad(filtered_angles)

    mean_cos = np.mean(np.cos(theta))
    mean_sin = np.mean(np.sin(theta))

    R = np.sqrt(mean_cos**2 + mean_sin**2)

    dci = 1 - R

    return float(dci)