import numpy as np


def compute_cri(cpi_history, window_size=5):
    """
    Compute Crowd Risk Index using temporal persistence.

    CRI is the moving average of recent CPI values.

    Parameters:
        cpi_history : list or array of previous CPI values
        window_size : number of recent frames to consider

    Returns:
        CRI value between 0 and 1
    """

    if cpi_history is None or len(cpi_history) == 0:
        raise ValueError("CPI history cannot be empty.")

    if window_size <= 0:
        raise ValueError("Window size must be greater than zero.")

    # Convert to numpy array
    cpi_history = np.asarray(cpi_history, dtype=np.float32)

    # Use only the latest window
    recent_cpi = cpi_history[-window_size:]

    # Temporal average
    cri = np.mean(recent_cpi)

    return float(np.clip(cri, 0.0, 1.0))