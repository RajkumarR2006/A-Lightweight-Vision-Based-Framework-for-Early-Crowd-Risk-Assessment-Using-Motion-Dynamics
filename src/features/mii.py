"""
mii.py
----------------------------------------------------
Motion Instability Index (MII)

This module computes the Motion Instability Index (MII)
using the standard deviation of the optical flow
magnitude matrix.
"""

import numpy as np


def compute_mii(magnitude: np.ndarray) -> float:
    """
    Compute Motion Instability Index (MII).

    Parameters
    ----------
    magnitude : np.ndarray
        Optical flow magnitude matrix.

    Returns
    -------
    float
        Motion Instability Index.
    """

    # Input validation
    if magnitude is None:
        raise ValueError("Magnitude matrix cannot be None.")

    if magnitude.size == 0:
        raise ValueError("Magnitude matrix is empty.")

    # Convert to float32
    magnitude = magnitude.astype(np.float32)

    # Standard deviation of motion magnitude
    mii = np.std(magnitude)

    return float(mii)