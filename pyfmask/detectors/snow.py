from typing import Optional

import numpy as np


def detect_snow(
    ndsi: np.ndarray,
    nir: np.ndarray,
    green: np.ndarray,
    bt: Optional[np.ndarray] = None,
    celsius_limit: int = 1000,
    ndsi_limit: float = 0.15,
    nir_limit: float = 1100,
    green_limit: int = 1000,
) -> np.ndarray:

    snow: np.ndarray = (ndsi > ndsi_limit) * (nir > nir_limit) * (green > green_limit)

    if bt:
        snow = (snow == True) & (bt < celsius_limit)

    return snow
