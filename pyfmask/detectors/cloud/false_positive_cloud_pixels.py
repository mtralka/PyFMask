from typing import Dict

from typing import Optional

import numpy as np
from pyfmask.utils.classes import DEMData
from pyfmask.utils.classes import SensorData

from skimage import morphology
from skimage.filters import threshold_otsu

np.seterr(divide="ignore")


def detect_false_positive_cloud_pixels(
    band_data: Dict[str, np.ndarray],
    ndbi: np.ndarray,
    ndvi: np.ndarray,
    platform_data: SensorData,
    snow: np.ndarray,
    water: np.ndarray,
    cloud: np.ndarray,
    cdi: Optional[np.ndarray],
    dem_data: Optional[DEMData],
):
    """Detect potential false positive urban, coastline, and snow/ice in cloud layer"""

    bt: Optional[np.ndarray] = band_data.get("BT", None)
    slope: Optional[np.ndarray] = dem_data.slope if dem_data is not None else None
    nodata_mask: np.ndarray = platform_data.nodata_mask
    out_resolution: int = platform_data.out_resolution

    ##
    # Urban areas
    ##
    potential_false_positives: np.ndarray = (
        (ndbi > 0) & (ndbi > ndvi) & (nodata_mask == False) & (water == False)
    )

    if np.sum(potential_false_positives == True) > 0:

        if bt is not None:

            # use Otsu methods
            bt_potential_false_positives = bt[
                (potential_false_positives == True) | (cloud == True)
            ]

            t = threshold_otsu(bt_potential_false_positives)

            tmp = bt_potential_false_positives[bt_potential_false_positives > t]

            if np.size(tmp) > 0:
                minimum_bt_temp = np.min(tmp)
                potential_false_positives = np.where(
                    bt < minimum_bt_temp, 0, potential_false_positives
                )

        # Sentinel-2
        if cdi is not None:
            # For Sentinel-2
            # Original threshold is -0.5: <-0.5 --> cloud, >-0.5 --> bright target
            # When threshold decreased, cloud overdetection potentially decrease,
            # but cloud missing increase
            potential_false_positives = np.where(
                cdi < -0.8, 0, potential_false_positives
            )

    ##
    # Add potential snow/ice pixels in mountain
    # 20 is from Burbank D W, Leland J, Fielding E, et al.
    # - Bedrock incision, rock uplift and threshold hillslopes in the northwestern Himalayas[J]. Nature, 1996, 379(6565): 505.
    ##
    if slope is not None:
        psnow_mountain = (snow == True) & (slope > 20)
        potential_false_positives = (potential_false_positives == True) | (
            psnow_mountain == True
        )

    ##
    # Buffer urban pixels with 500m window
    ##
    width_m = 250
    width_px = int(width_m / out_resolution)
    potential_false_positives = morphology.binary_dilation(
        potential_false_positives.astype(bool), morphology.square(2 * width_px + 1)
    )

    potential_false_positives = (potential_false_positives == True) | (snow == True)
    potential_false_positives = (potential_false_positives == True) & (
        nodata_mask == False
    )

    return potential_false_positives
