import numpy as np


def min_max_normalize(value, minimum, maximum):
    """
    Normalize a value to [0, 1].
    """

    if maximum <= minimum:
        raise ValueError("Maximum must be greater than minimum.")

    normalized = (value - minimum) / (maximum - minimum)

    return float(np.clip(normalized, 0.0, 1.0))


def compute_cpi(
    density,
    mii,
    dci,
    density_min,
    density_max,
    mii_min,
    mii_max,
    motion_weight=0.5,
    direction_weight=0.5
):
    """
    Compute Congestion Pressure Index.

    CPI = Normalized Density *
          (motion_weight * Normalized MII +
           direction_weight * DCI)
    """

    if not 0 <= dci <= 1:
        raise ValueError("DCI must be between 0 and 1.")

    if motion_weight < 0 or direction_weight < 0:
        raise ValueError("Weights must be non-negative.")

    weight_sum = motion_weight + direction_weight

    if weight_sum == 0:
        raise ValueError("At least one weight must be greater than zero.")

    # Normalize weights
    motion_weight /= weight_sum
    direction_weight /= weight_sum

    # Normalize density
    normalized_density = min_max_normalize(
        density,
        density_min,
        density_max
    )

    # Normalize MII
    normalized_mii = min_max_normalize(
        mii,
        mii_min,
        mii_max
    )

    # Calculate CPI
    cpi = normalized_density * (
        motion_weight * normalized_mii +
        direction_weight * dci
    )

    return float(np.clip(cpi, 0.0, 1.0))