from typing import Optional

import numpy as np


def detect_water(
    nir: np.ndarray,
    ndvi: np.ndarray,
    nodata_mask: np.ndarray,
    snow: Optional[np.ndarray] = None,
    gswo: Optional[np.ndarray] = None,
    primary_ndvi_limit: float = 0.01,
    secondary_ndvi_limit: float = 0.1,
    primary_nir_limit: float = 1100.0,
    secondary_nir_limit: float = 500.0,
) -> np.ndarray:

    water: np.ndarray = np.zeros(nir.shape).astype(bool)

    water = np.where(
        ((ndvi < primary_ndvi_limit) & (nir < primary_nir_limit))
        | ((ndvi < secondary_ndvi_limit) & (ndvi > 0) & (nir < secondary_nir_limit)),
        1,
        water,
    )

    water = np.where(nodata_mask, 0, water)

    all_water: np.ndarray = np.where(nodata_mask, 0, water)

    if not gswo or not snow:
        return water

    # if water present
    if np.sum(gswo) > 0:
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

        print(f"gswater_occur={gswater_occur}")
        if gswater_occur < 0:
            return water

        water_gs = gswo > gswater_occur
        all_water = np.where(water_gs == True, 1, all_water)

        water = np.where((water_gs == True) & (snow == False), 1, water)

        water = np.where(nodata_mask, 0, water)
        all_water = np.where(nodata_mask, 0, all_water)

    return water
