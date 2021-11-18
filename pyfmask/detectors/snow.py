from typing import Dict
from typing import Optional

import numpy as np


def detect_snow(
    ndsi: np.ndarray,
    band_data: Dict[str, np.ndarray],
    celsius_limit: int = 1000,
    ndsi_limit: float = 0.15,
    nir_limit: float = 1100,
    green_limit: int = 1000,
) -> np.ndarray:

    nir: np.ndarray = band_data["NIR"]
    green: np.ndarray = band_data["GREEN"]
    bt: Optional[np.ndarray] = band_data.get("BT")

    snow: np.ndarray = (ndsi > ndsi_limit) * (nir > nir_limit) * (green > green_limit)

    if bt is not None:
        snow = (snow == True) & (bt < celsius_limit)

    return snow
