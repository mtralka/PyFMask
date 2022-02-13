from typing import Dict

import numpy as np
from scipy.ndimage import uniform_filter
import logging.config

logger = logging.getLogger(__name__)


def detect_absolute_snow(
    sensor: str,
    detected_snow: np.ndarray,
    band_data: Dict[str, np.ndarray],
    vis_saturation: np.ndarray,
    ndsi: np.ndarray,
) -> np.ndarray:

    # L08_OLI - 10km @ 30m
    # S2_MSI - 10km @ 20m
    SCSI10_index_dict: Dict[str, int] = {"L08_OLI": 333, "S2_MSI": 501}

    window_size: int
    for platform, value in SCSI10_index_dict.items():
        if sensor == platform:
            window_size = value
            break
    else:
        raise ValueError("`sensor` does not contain SCSI10 value")

    green: np.ndarray = np.array(band_data["GREEN"], copy=True).astype(np.float32)

    green[green < 0] = 0

    array_1: np.ndarray = uniform_filter(green, window_size, mode="constant", cval=0)
    array_2: np.ndarray = uniform_filter(
        green * green, window_size, mode="constant", cval=0
    )

    mask: np.ndarray = np.where(green != 0, 1, 0)
    weight: np.ndarray = uniform_filter(
        mask.astype(np.float32), window_size, mode="constant", cval=0
    )

    array_1 = np.where(weight > 0, array_1 / (weight + 1e-7), 0)
    array_2 = np.where(weight > 0, array_2 / (weight + 1e-7), 0)

    scsi: np.ndarray = array_2 - array_1 * array_1
    scsi[(scsi <= 0) | (mask == 0)] = 0
    scsi = np.sqrt(scsi)

    scsi = scsi * (1 - ndsi)
    absolute_snow: np.ndarray = (
        (scsi < 9) & (detected_snow == True) & (vis_saturation == False)
    )

    logger.debug("Detected %s pixels of absolute snow", np.sum(absolute_snow))
    return absolute_snow
