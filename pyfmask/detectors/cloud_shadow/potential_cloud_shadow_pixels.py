from typing import Optional
from typing import Tuple

import numpy as np
from pyfmask.detectors.cloud_shadow.match_cloud_shadows import shadow
from pyfmask.classes import DEMData
from pyfmask.classes import PlatformData
from skimage import morphology
import logging.config

logger = logging.getLogger(__name__)


def detect_potential_cloud_shadow_pixels(
    platform_data: PlatformData,
    dem_data: Optional[DEMData],
    clear_land: np.ndarray,
    low_percent: float = 0.175,
    potential_shadow_threshold: int = 500,
) -> np.ndarray:
    """Detect potential cloud shadow pixels"""

    # for evential use with topo correction
    sun_zenith: float = 90.0 - platform_data.sun_elevation
    sun_azimuth: float = platform_data.sun_azimuth

    slope: Optional[np.ndarray] = dem_data.slope if dem_data else None
    aspect: Optional[np.ndarray] = dem_data.aspect if dem_data else None

    nodata_mask: np.ndarray = platform_data.nodata_mask

    nir: np.ndarray = platform_data.band_data["NIR"]
    swir1: np.ndarray = platform_data.band_data["SWIR1"]

    nir_corrected: np.ndarray = platform_data.band_data["NIR"]
    swir1_corrected: np.ndarray = platform_data.band_data["SWIR1"]

    if (slope is not None) & (aspect is not None):
        nir_corrected, swir1_corrected = get_topo_corrected_bands(nir, swir1)

    ##
    # Flood fill `clear_land` in `nir_corrected` and `swir1_corrected`
    ##
    nir_background: np.ndarray = np.percentile(nir[clear_land], 100 * low_percent)
    swir1_background: np.ndarray = np.percentile(swir1[clear_land], 100 * low_percent)

    nir_corrected = np.where(
        (nodata_mask) | (np.isnan(nir_corrected)), nir_background, nir_corrected
    )
    nir_corrected_filled: np.ndarray = imfill_skimage(nir_corrected.astype(np.float32))
    nir_corrected_difference: np.ndarray = nir_corrected_filled - nir_corrected

    swir1_corrected = np.where(
        (nodata_mask) | (np.isnan(swir1_corrected)), swir1_background, swir1_corrected
    )
    swir1_corrected_filled: np.ndarray = imfill_skimage(
        swir1_corrected.astype(np.float32)
    )
    swir1_corrected_difference: np.ndarray = swir1_corrected_filled - swir1_corrected

    ##
    # Compute shadow probability
    ##
    shadow_probability: np.ndarray = np.minimum(
        nir_corrected_difference, swir1_corrected_difference
    )

    ##
    # Potential shadow mask
    ##
    shadow_mask: np.ndarray = np.where(
        shadow_probability > potential_shadow_threshold, 1, 0
    ).astype(np.uint8)

    ##
    # Remove potential shadows smaller than 3 pixels
    ##
    shadow_mask = morphology.remove_small_objects(
        shadow_mask.astype(bool), 3, connectivity=2
    )
    shadow_mask = np.where(nodata_mask, 255, shadow_mask)

    logger.debug("Sum of cloud shadow mask %s", np.sum(shadow_mask))

    return shadow_mask


def get_topo_corrected_bands(
    nir: np.ndarray, swir1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Topo correct `nir` and `swir`"""

    # TODO

    return nir, swir1


def imfill_skimage(array: np.ndarray) -> np.ndarray:
    """
    Replicates the imfill function available within MATLAB. Based on the
    example provided in
    https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html
    """

    seed: np.ndarray = array.copy()

    ##
    # Define seed and start points for erosion
    ##
    seed[1:-1, 1:-1] = array.max()

    ##
    # Fill holes
    ##
    filled_array: np.ndarray = morphology.reconstruction(seed, array, method="erosion")

    return filled_array
