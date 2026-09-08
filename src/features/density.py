import numpy as np


def compute_density(foreground_mask):
    """
    Compute normalized foreground occupancy.

    Density = foreground pixels / total pixels
    """

    if foreground_mask is None:
        raise ValueError("Foreground mask cannot be None.")

    if foreground_mask.size == 0:
        raise ValueError("Foreground mask is empty.")

    binary_mask = foreground_mask > 0

    foreground_pixels = np.count_nonzero(binary_mask)

    total_pixels = binary_mask.size

    density = foreground_pixels / total_pixels

    return float(density)


def normalize_density(
    density,
    low=0.0025,
    high=0.0617
):
    """
    Robust percentile-based density normalization.

    Values below low are mapped to 0.
    Values above high are mapped to 1.
    """

    if high <= low:
        raise ValueError("High percentile must be greater than low percentile.")

    normalized = (density - low) / (high - low)

    return float(np.clip(normalized, 0.0, 1.0))