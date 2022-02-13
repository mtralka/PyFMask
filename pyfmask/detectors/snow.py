from typing import Dict
from typing import Optional

import numpy as np
import logging.config

logger = logging.getLogger(__name__)


def detect_snow(ndsi: np.ndarray, band_data: Dict[str, np.ndarray]) -> np.ndarray:

    nir: np.ndarray = band_data["NIR"]
    green: np.ndarray = band_data["GREEN"]
    bt: Optional[np.ndarray] = band_data.get("BT")

    snow: np.ndarray = (ndsi > 0.15) & (nir > 1100) & (green > 1000)

    if bt is not None:
        snow = (snow == True) & (bt < 1000)

    logger.debug("Detected %s pixels of snow", np.sum(snow))
    return snow
