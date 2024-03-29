from typing import Optional, Tuple
from pyfmask.classes import GSWOData
import numpy as np
import logging.config

logger = logging.getLogger(__name__)


def detect_water(
    nir: np.ndarray,
    ndvi: np.ndarray,
    nodata_mask: np.ndarray,
    snow: Optional[np.ndarray] = None,
    gswo_data: Optional[GSWOData] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    water: np.ndarray = np.zeros(nir.shape).astype(bool)
    gswo: Optional[np.ndarray] = gswo_data.gswo if gswo_data is not None else None

    water = np.where(
        ((ndvi < 0.01) & (nir < 1100)) | ((ndvi < 0.1) & (ndvi > 0) & (nir < 500)),
        1,
        water,
    )

    water = np.where(nodata_mask, 0, water)

    all_water: np.ndarray = np.where(nodata_mask, 0, water)

    if gswo is None or snow is None:
        return water, all_water

    # if not water present
    if np.sum(gswo) <= 0:
        return water, all_water

    # assume the water occurances are similar in each whole scene
    # global surface water occurance (GSWO)
    # low level to exclude the commssion errors as water.
    # 5% tolerances
    mm: np.ndarray = water[water == 1]
    if np.sum(mm) > 0:
        gswater_occur = (
            np.percentile(gswo[water == 1], 17.5) - 5
        )  # prctile(gswater(water==1),17.5)-5
    else:
        gswater_occur = 90

    # If > 90, we want to use at least 90%
    if gswater_occur > 90:
        gswater_occur = 90

    logger.debug("Gswater occur, %s", gswater_occur)

    if gswater_occur < 0:
        return water, all_water

    water_gs = gswo > gswater_occur
    all_water = np.where(water_gs == True, 1, all_water)

    water = np.where((water_gs == True) & (snow == False), 1, water)

    water = np.where(nodata_mask, 0, water)
    all_water = np.where(nodata_mask, 0, all_water)

    logger.debug("Detected %s pixels of water", np.sum(water))

    return water, all_water
