from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from pyfmask.utils.classes import DEMData
from pyfmask.utils.classes import PotentialCloudPixels


def detect_probable_cloud_pixels(
    ndsi: np.ndarray,
    ndvi: np.ndarray,
    band_data: Dict[str, np.ndarray],
    vis_saturation: np.ndarray,
    dem_data: Optional[DEMData] = None,
    nodata_mask: Optional[np.ndarray] = None,
    ndsi_limit: float = 0.8,
    ndvi_limit: float = 0.8,
    swir2_limit: float = 300.0,
    bt_limt: float = 2700.0,
    whiteness_limit: float = 0.7,
    ratio_nir_swir_limit: float = 0.75,
    normalized_cirrus_limit: int = 100,
) -> PotentialCloudPixels:

    nir: np.ndarray = band_data["NIR"]
    swir1: np.ndarray = band_data["SWIR1"]
    swir2: np.ndarray = band_data["SWIR2"]
    blue: np.ndarray = band_data["BLUE"]
    green: np.ndarray = band_data["GREEN"]
    red: np.ndarray = band_data["RED"]
    cirrus: Optional[np.ndarray] = band_data.get("CIRRUS")
    bt: Optional[np.ndarray] = band_data.get("BT")

    dem: Optional[np.ndarray] = dem_data.dem if dem_data is not None else None

    ##
    # Step 1: Basic cloud test
    ##
    potential_pixels: np.ndarray = (
        (ndsi < ndsi_limit) & (ndvi < ndvi_limit) & (swir2 > swir2_limit)
    )

    if bt is not None:
        potential_pixels = (potential_pixels == True) & (bt < bt_limt)

    ##
    # Step 2: Whiteness test
    ##
    visible_mean: np.ndarray = (blue + green + red) / 3.0
    whiteness: np.ndarray = (
        np.absolute(blue - visible_mean)
        + np.absolute(green - visible_mean)
        + np.absolute(red - visible_mean)
    ) / visible_mean
    whiteness[vis_saturation == True] = 0
    # If one visible is saturated whiteness == 0
    potential_pixels = (potential_pixels == True) & (whiteness < whiteness_limit)

    ##
    # Step 3: Haze test
    ##
    hot: np.ndarray = blue - 0.5 * red - 800
    potential_pixels = (potential_pixels == True) & (
        (hot > 0) | (vis_saturation == True)
    )

    ##
    # Step 4: Ratio NIR / SWIR > limit
    ##
    ratio_nir_swir: np.ndarray = nir / swir1
    potential_pixels = (potential_pixels == True) & (
        ratio_nir_swir > ratio_nir_swir_limit
    )

    ##
    # Optional Step 5: Ratio NIR / SWIR > limit
    ##
    normalized_cirrus: Optional[np.ndarray] = None
    if cirrus is not None:
        normalized_cirrus = normalize_cirrus_dem(
            potential_pixels=potential_pixels,
            cirrus=cirrus,
            nodata_mask=nodata_mask,
            dem=dem,
        )

        potential_pixels = (potential_pixels == True) | (
            normalized_cirrus > normalized_cirrus_limit
        )

    data: PotentialCloudPixels = PotentialCloudPixels(
        potential_pixels=potential_pixels,
        normalized_cirrus=normalized_cirrus,
        whiteness=whiteness,
        hot=hot,
    )
    return data


def normalize_cirrus_dem(
    potential_pixels: np.ndarray,
    cirrus: np.ndarray,
    nodata_mask: Optional[np.ndarray],
    dem: Optional[np.ndarray] = None,
    percentile: Union[float, int] = 2,
) -> np.ndarray:

    normalized_cirrus: np.ndarray = np.zeros(cirrus.shape).astype(np.float32)

    # clear sky pixels and valid data
    valid_clear_sky: np.ndarray = (potential_pixels == False) & (nodata_mask == False)

    if dem is None or np.all(dem == dem[0]):
        # this version with no DEM
        prcnt = np.percentile(cirrus[valid_clear_sky], percentile)
        # taking over data area
        normalized_cirrus = np.where(
            nodata_mask == False, cirrus - prcnt, normalized_cirrus
        )
        normalized_cirrus[normalized_cirrus < 0] = 0
        return normalized_cirrus

    # with DEM

    # Taking percentile to remove outliers from DEM
    dem_start: int = int(np.percentile(dem[dem != -9999], 0.001))
    dem_end: int = int(np.percentile(dem[dem != -9999], 99.999))
    step: int = 100

    cirrus_lowest: Union[float, int] = 0.0
    for k in np.arange(dem_start, dem_end + step, step):
        # take only in the range and only clear
        mm: np.ndarray = (cirrus >= k) & (cirrus < (k + step))
        mm_clear: np.ndarray = (mm == True) & (valid_clear_sky == True)
        if np.sum(mm_clear) > 0:
            cirrus_lowest = np.percentile(cirrus[mm_clear], percentile)
        normalized_cirrus = np.where(
            (nodata_mask == False) & (mm == True),
            cirrus - cirrus_lowest,
            normalized_cirrus,
        )

    normalized_cirrus[normalized_cirrus < 0] = 0

    return normalized_cirrus
