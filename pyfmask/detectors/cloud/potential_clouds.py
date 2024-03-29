import logging.config
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
from pyfmask.classes import DEMData
from pyfmask.classes import PotentialCloudPixels
from pyfmask.classes import PotentialClouds
import statsmodels.api as sm


logger = logging.getLogger(__name__)


def detect_potential_clouds(
    band_data: Dict[str, np.ndarray],
    dem_data: Optional[DEMData],
    potential_cloud_pixels: PotentialCloudPixels,
    nodata_mask: np.ndarray,
    water: np.ndarray,
    thin_cirrus_weight: Union[int, float],
    cloud_probability_threshold: Union[int, float],
    ndsi: np.ndarray,
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    vis_saturation: np.ndarray,
    clear_pixels_threshold: int = 40000,
    low_percent: float = 0.175,
    high_percent: float = 0.825,
):

    nir: np.ndarray = band_data["NIR"]
    swir1: np.ndarray = band_data["SWIR1"]
    cirrus: Optional[np.ndarray] = band_data.get("CIRRUS", None)
    bt: Optional[np.ndarray] = band_data.get("BT", None)
    dem: Optional[np.ndarray] = dem_data.dem if dem_data is not None else None

    potential_clouds = np.zeros(nir.shape, dtype=np.uint8)

    idused: np.ndarray = np.zeros(nir.shape)

    bt_normalized_dem: Optional[np.ndarray] = None

    clear_pixels: np.ndarray = (potential_cloud_pixels.potential_pixels == False) & (
        nodata_mask == False
    )
    sum_clear_pixels: int = np.sum(clear_pixels == True)

    clear_land_mask: np.ndarray = (clear_pixels == True) & (water == False)
    clear_water_mask: np.ndarray = (clear_pixels == True) & (water == True)

    temp_test_low: Union[int, float] = 0
    temp_test_high: Union[int, float] = 0

    ##
    # If potential cloud pixels are less than `clear_pixels_threshold`,
    # remove all potential cloud pixels and return
    ##
    if sum_clear_pixels <= clear_pixels_threshold:
        potential_clouds = np.where(clear_pixels == True, 1, potential_clouds)
        potential_clouds = np.where(nodata_mask == True, 0, potential_clouds)

        # cloud_over_land_probability & cloud_over_water_probability are both 100
        over_land_water_probability: np.ndarray = 100 * np.ones(
            potential_clouds.shape
        ).astype(np.uint8)

        return PotentialClouds(
            sum_clear_pixels=sum_clear_pixels,
            cloud=potential_clouds,
            clear_land=idused,
            over_water_probability=over_land_water_probability,
            over_land_probability=over_land_water_probability,
        )

    ##
    # Calculate probability of thin clouds
    ##
    probability_thin_cloud: Union[int, np.ndarray] = 0
    if cirrus is not None:
        probability_thin_cloud = cirrus / 400
        probability_thin_cloud = np.where(
            probability_thin_cloud < 0, 0, probability_thin_cloud
        )

    ##
    # Cloud probability over land
    ##
    over_land_probability_indicator: Union[float, int] = (
        100.0 * np.sum(clear_land_mask == True) / np.sum(nodata_mask == False)
    )
    if over_land_probability_indicator >= 0.1:
        idused = clear_land_mask
    else:
        idused = clear_pixels

    land_probability_temperature: Union[int, np.ndarray, float] = 1
    land_probability_brightness: Union[int, np.ndarray, float] = 1

    # if BT is available, use temperature probability
    if bt is not None and dem is not None:

        bt_normalized_dem = normalize_bt(bt, dem, idused, low_percent, high_percent)

        (
            land_probability_temperature,
            temp_test_low,
            temp_test_high,
        ) = land_temperature_probability(
            cast(np.ndarray, bt_normalized_dem), idused, low_percent, high_percent
        )

    # if BT is not available, use HOT probability
    else:

        land_probability_brightness = land_brightness_probability_hot(
            potential_cloud_pixels.hot, idused, low_percent, high_percent
        )

        # land_probability_brightness = probl_brightness(
        #     potential_cloud_pixels.hot, idused, low_percent, high_percent
        # )

    over_land_probability_variance: np.ndarray = spectral_variance_probability(
        ndsi, ndvi, ndbi, vis_saturation, potential_cloud_pixels.whiteness
    )

    over_land_probability: np.ndarray = (
        land_probability_temperature
        * over_land_probability_variance
        * land_probability_brightness
        + thin_cirrus_weight * probability_thin_cloud
    )
    over_land_probability = 100.0 * over_land_probability

    ##
    # Cloud probability over water
    ##
    over_water_probability_temperature: Union[int, np.ndarray] = 1
    if (bt is not None) & (np.sum(clear_water_mask == True) > 100):
        over_water_probability_temperature = water_temperature_probability(
            cast(np.ndarray, bt), clear_water_mask, high_percent
        )

    over_water_probability_brightness: np.ndarray = water_brightness_probability(swir1)

    over_water_probability: np.ndarray = (
        over_water_probability_temperature * over_water_probability_brightness
        + thin_cirrus_weight * probability_thin_cloud
    )
    over_water_probability = 100.0 * over_water_probability

    if np.sum(clear_water_mask == True) > 0:
        wclr_h = np.percentile(
            over_water_probability[clear_water_mask == True], 100 * high_percent
        )
    else:
        wclr_h = 0

    clr_h: Union[int, float] = 0
    if np.sum(clear_land_mask == True) > 0:
        clr_h = np.percentile(
            over_land_probability[clear_land_mask == True], 100 * high_percent
        )

    dynamic_water_max: Union[int, float] = (
        wclr_h + cloud_probability_threshold
    )  # dynamic threshold (water)
    dynamic_land_max: Union[int, float] = (
        clr_h + cloud_probability_threshold
    )  # dynamic threshold (land)

    id_final_cld = (potential_cloud_pixels.potential_pixels == True) & (
        ((over_land_probability > dynamic_land_max) & (water == False))
        | ((over_water_probability > dynamic_water_max) & (water == True))
    )

    ##
    # Handle extremely cold clouds
    ##
    if bt is not None and dem is not None:
        id_final_cld = (id_final_cld == True) | (
            cast(np.ndarray, bt_normalized_dem) < (temp_test_low - 3500)
        )

    ##
    # Assign potential clouds
    ##
    potential_clouds = np.where(id_final_cld, 1, potential_clouds)
    potential_clouds = np.where(nodata_mask == True, 0, potential_clouds)

    logger.debug("%s potential clouds", np.sum(potential_clouds))

    return PotentialClouds(
        sum_clear_pixels=sum_clear_pixels,
        cloud=potential_clouds,
        temp_test_low=temp_test_low,
        temp_test_high=temp_test_high,
        clear_land=idused,
        bt_normalized_dem=bt_normalized_dem,
        over_land_probability=over_land_probability,
        over_water_probability=over_water_probability,
    )


def normalize_bt(bt, dem, idused, low_percent, high_percent):
    # Normalizing temperature over DEM
    # a linear model used for normalization (qui et al., 2017 RSE)
    norm_bt = np.array(bt, copy=True)
    dem_mask = (dem != -9999) & (bt != -9999)

    if np.sum(dem_mask) < 100:
        return norm_bt

    dem_b = np.percentile(dem[dem_mask], 0.0001)
    dem_t = np.percentile(dem[dem_mask], 99.999)  # % further exclude non dem pixels.
    # array of temp pixel over clear land (idused)
    temp_cl = bt[idused]
    temp_min = np.percentile(temp_cl, low_percent * 100)
    temp_max = np.percentile(temp_cl, high_percent * 100)

    # making a mask of valid observations along with DEM
    mm = (bt > temp_min) & (bt < temp_max) & (idused == True) & (dem_mask == True)
    data_bt_c_clear = bt[mm].astype(np.float32)
    data_dem_clear = dem[mm].astype(np.float32)
    total_sample = 40000  # selecting num points with stratification

    # Performaing stratification
    step = 300
    num_strata_avail = 0
    for k in np.arange(dem_b, dem_t + step, step):
        mm = (data_dem_clear >= k) & (data_dem_clear < (k + step))
        if np.sum(mm) > 0:
            num_strata_avail = num_strata_avail + 1
    num_per_strata = int(round(total_sample / num_strata_avail, 0))
    if num_per_strata < 1:
        # meanining not enough points: return original BT
        return norm_bt
    else:
        dem_sampled = np.array([])
        bt_sampled = np.array([])
        for k in np.arange(dem_b, dem_t + step, step):
            mm = (data_dem_clear >= k) & (data_dem_clear < (k + step))
            if np.sum(mm) > 0:
                tmp_dem = data_dem_clear[mm]
                tmp_bt = data_bt_c_clear[mm]
                # randomly selecting locations
                loc_random = np.random.choice(
                    np.arange(0, tmp_dem.shape[0]),
                    size=min(tmp_dem.shape[0], num_per_strata),
                    replace=False,
                )
                dem_sampled = np.concatenate((dem_sampled, tmp_dem[loc_random]))
                bt_sampled = np.concatenate((bt_sampled, tmp_bt[loc_random]))
        n_samples = dem_sampled.shape[0]

        # now performing regression
        X = sm.add_constant(dem_sampled)
        Y = bt_sampled
        est = sm.OLS(Y, X).fit()

        logger.debug("%s regression paramters - %s", len(est.params), est.params)

        if len(est.params) == 1:
            return norm_bt
        rate_lapse = est.params[1]
        rate_lapse_pvalue = est.pvalues[1]

        # only perform normalization when
        # rate_lapse<0 and its p-value is significant
        if (rate_lapse < 0) & (rate_lapse_pvalue < 0.05):
            norm_bt = np.where(dem_mask, bt - rate_lapse * (dem - dem_b), bt)

    return norm_bt


def water_temperature_probability(
    bt: np.ndarray, clear_water_mask: np.ndarray, high_percent: float
) -> np.ndarray:
    """Calculate temperature probability over water"""

    ##
    # Get BT for clear water pixels
    ##
    bt_of_clear_water: np.ndarray = bt[clear_water_mask]

    ##
    # Take percentile
    ##
    bt_percentile: np.ndarray = np.percentile(
        bt_of_clear_water, 100 * high_percent
    )  # (Eq. 8 - Zhu 2012)

    ##
    # Offset temperature and divide by 4degC
    ##
    probability: np.ndarray = (bt_percentile - bt) / 400  # Eq. 9 (Zhu 2012)
    probability = np.where(probability < 0, 0, probability)

    return probability


def water_brightness_probability(
    swir1: np.ndarray, temperature_brightness: int = 1100
) -> np.ndarray:
    """Calculate brightness probability over water"""

    probability: np.ndarray = swir1 / temperature_brightness  # (Eq. 10 - Zhu 2012)
    probability = np.where(probability > 1, 1, probability)
    probability = np.where(probability < 0, 0, probability)

    return probability


def land_temperature_probability(
    bt_normalized_dem: np.ndarray,
    idused: np.ndarray,
    low_percent: float,
    high_percent: float,
) -> Tuple[np.ndarray, Union[int, float], Union[int, float]]:
    """Calculate land temperature probability"""

    over_clear_land_pixels: np.ndarray = bt_normalized_dem[idused]

    temp_buffer: int = 4 * 100

    low_percentile: Union[int, float] = np.percentile(
        over_clear_land_pixels, 100 * low_percent
    )  # Eq. 12-13 (Zhu 2012)
    high_percentile: Union[int, float] = np.percentile(
        over_clear_land_pixels, 100 * high_percent
    )  # Eq. 12-13 (Zhu 2012)

    temp_test_low: Union[int, float] = low_percentile - temp_buffer
    temp_test_high: Union[int, float] = high_percentile + temp_buffer

    logger.debug("Cloud temp test low: %s ", temp_test_low)
    logger.debug("Cloud temp test high: %s ", temp_test_high)

    temp_limit: Union[int, float] = temp_test_high - temp_test_low

    probability_temperature = (
        temp_test_high - bt_normalized_dem
    ) / temp_limit  # Eq. 14 (Zhu 2012)

    # probability_temp (i)(n) can be higher than 1
    probability_temperature = np.where(
        probability_temperature < 0, 0, probability_temperature
    )

    return probability_temperature, temp_test_low, temp_test_high


def land_brightness_probability_hot(
    hot: np.ndarray, idused: np.ndarray, low_percent: float, high_percent: float
) -> np.ndarray:
    """Calculate land brightness probability using HOT"""

    over_clear_land_pixels: np.ndarray = hot[idused]

    low_hot_percentile: np.ndarray = (
        np.percentile(over_clear_land_pixels, 100 * low_percent) - 400
    )
    high_hot_percentile: np.ndarray = (
        np.percentile(over_clear_land_pixels, 100 * high_percent) + 400
    )

    probability_brightness: np.ndarray = (hot - low_hot_percentile) / (
        high_hot_percentile - low_hot_percentile
    )
    probability_brightness = np.where(
        probability_brightness < 0, 0, probability_brightness
    )

    # probability_brightness(i)(n) cannot be higher than 1
    probability_brightness = np.where(
        probability_brightness > 1, 1, probability_brightness
    )

    return probability_brightness


def spectral_variance_probability(
    ndsi: np.ndarray,
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    vis_saturation: np.ndarray,
    whiteness: np.ndarray,
) -> np.ndarray:
    """Overland spectral variance test"""

    ndsi = np.where((vis_saturation == True) & (ndsi < 0), 0, ndsi)
    ndvi = np.where((vis_saturation == True) & (ndvi > 0), 0, ndvi)

    ndsi = np.absolute(ndsi)
    ndvi = np.absolute(ndvi)
    ndbi = np.absolute(ndbi)

    probability_variance: np.ndarray = 1 - np.maximum(
        np.maximum(np.maximum(ndsi, ndvi), ndbi), whiteness
    )  # Eq. 15 with added NDBI (Zhe 2012)

    return probability_variance


# def normalize_bt(
#     bt: np.ndarray,
#     dem: np.ndarray,
#     idused: np.ndarray,
#     low_percent: float,
#     high_percent: float,
#     stratification_step: int = 300,
# ) -> np.ndarray:
#     """Normalize temperature over DEM using linear model (qui et al., 2017 RSE)"""

#     normalized_bt: np.ndarray = np.array(bt, copy=True)

#     masked_dem: np.ndarray = (dem != -9999) & (bt != -9999)

#     if np.sum(masked_dem) < 100:
#         return normalized_bt

#     dem_b = np.percentile(dem[masked_dem], 0.0001)
#     dem_t = np.percentile(dem[masked_dem], 99.999)  # % further exclude non dem pixels.

#     # array of temp pixel over clear land (idused)
#     temp_cl = bt[idused]
#     temp_min = np.percentile(temp_cl, low_percent * 100)
#     temp_max = np.percentile(temp_cl, high_percent * 100)

#     # making a mask of valid observations along with DEM
#     mm = (bt > temp_min) & (bt < temp_max) & (idused == True) & (masked_dem == True)
#     data_bt_c_clear = bt[mm].astype(np.float32)
#     data_dem_clear = dem[mm].astype(np.float32)
#     total_sample = 40000  # selecting num points with stratification

#     ##
#     # Performing stratification
#     ##
#     strata_available: int = 0
#     for k in np.arange(dem_b, dem_t + stratification_step, stratification_step):
#         mm = (data_dem_clear >= k) & (data_dem_clear < (k + stratification_step))
#         if np.sum(mm) > 0:
#             strata_available = strata_available + 1
#     points_per_strata: Union[int, float] = int(
#         round(total_sample / strata_available, 0)
#     )

#     # not enough points for normalization
#     if points_per_strata < 1:
#         # return BT
#         return normalized_bt

#     dem_sampled = np.array([])
#     bt_sampled = np.array([])
#     for k in np.arange(dem_b, dem_t + stratification_step, stratification_step):
#         mm = (data_dem_clear >= k) & (data_dem_clear < (k + stratification_step))
#         if np.sum(mm) > 0:
#             tmp_dem = data_dem_clear[mm]
#             tmp_bt = data_bt_c_clear[mm]
#             # randomly selecting locations
#             loc_random = np.random.choice(
#                 np.arange(0, tmp_dem.shape[0]),
#                 size=min(tmp_dem.shape[0], points_per_strata),
#                 replace=False,
#             )
#             dem_sampled = np.concatenate((dem_sampled, tmp_dem[loc_random]))
#             bt_sampled = np.concatenate((bt_sampled, tmp_bt[loc_random]))
#     n_samples = dem_sampled.shape[0]

#     ##
#     # Regress
#     ##
#     X = sm.add_constant(dem_sampled)
#     Y = bt_sampled
#     est = sm.OLS(Y, X).fit()


#     rate_lapse = est.params[1]
#     rate_lapse_pvalue = est.pvalues[1]

#     # only perform normalization if
#     # rate_lapse < 0 and corresponding p-value is significant
#     if (rate_lapse < 0) & (rate_lapse_pvalue < 0.05):
#         norm_bt = np.where(normalized_bt, bt - rate_lapse * (dem - dem_b), bt)

#     return norm_bt
