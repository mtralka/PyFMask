import logging.config
import math
from typing import Optional
from typing import Union

import numpy as np
from pyfmask.classes import DEMData
from pyfmask.classes import PlatformData
from skimage.measure import label
from skimage.measure import regionprops


####
# TODO
# - optimize matching
####

logger = logging.getLogger(__name__)


def shadow(H, sun_elev, sun_azim):  # in degrees
    L = H / math.tan(sun_elev * math.pi / 180.0)  # convert to radians
    dx = -L * math.sin((sun_azim) * math.pi / 180.0)  # convert to radians
    dy = L * math.cos((sun_azim) * math.pi / 180.0)  # convert to radians
    dx = int(dx)
    dy = int(dy)
    return dx, dy


def match_cloud_shadows(
    cloud: np.ndarray,
    sum_clear_pixels: int,
    all_water: np.ndarray,
    potential_cloud_shadow_pixels: np.ndarray,
    platform_data: PlatformData,
    dem_data: Optional[DEMData],
    temp_test_low: Union[int, float],
    temp_test_high: Union[int, float],
    neighbour_tolerance: float = 4.25,
    similarity_matched_threshold: float = 0.95,
    low_percent: float = 0.175,
    high_percent: float = 0.825,
    cloud_pixels_limit: int = 40000,
    min_cloud_height: Union[int, float] = 200.0,
    max_cloud_height: Union[int, float] = 12000.0,
) -> np.ndarray:

    ENVIRONMENTAL_LAPSE_RATE: float = 6.5  # degrees / km
    DRY_ADIABATIC_LAPSE_RATE: float = 9.8  # degrees / km

    min_cloud_height = float(min_cloud_height)
    max_cloud_height = float(max_cloud_height)

    dem: Optional[np.ndarray] = dem_data.dem if dem_data else None
    bt: Optional[np.ndarray] = platform_data.band_data.get("BT", None)

    out_resolution: int = platform_data.out_resolution
    nodata_mask: np.ndarray = platform_data.nodata_mask
    sun_elevation: float = platform_data.sun_elevation
    sun_azimuth: float = platform_data.sun_azimuth

    cloud_potential: np.ndarray = (cloud == True) & (nodata_mask == False)
    shadow_potential: np.ndarray = np.where(
        potential_cloud_shadow_pixels == True, True, False
    )

    # template array for matched cloud shadows
    matched_cloud_shadow_layer: np.ndarray = np.zeros(cloud.shape).astype(bool)

    revised_cloud_percent: np.ndarray = np.sum(cloud_potential == True) / np.sum(
        nodata_mask == False
    )

    ##
    # Don't matched shadows if there are too many clouds
    ##
    if (sum_clear_pixels <= cloud_pixels_limit) | (revised_cloud_percent >= 0.90):

        logger.debug("Skipping cloud shadow detection due to high cloud cover")

        matched_cloud_shadow_layer = np.where(
            shadow_potential == False, 1, matched_cloud_shadow_layer
        )
        similar_num = -1
        return matched_cloud_shadow_layer

    ##
    # Extract DEM data
    ##
    base_dem_height: Union[int, float] = 0
    if dem is not None:
        mm = (nodata_mask == False) & (dem != -9999)

        if np.sum(mm) > 0:
            base_dem_height = np.percentile(dem[mm], 0.001)

    ##
    # Label clouds
    ##
    cloud_labels: np.ndarray = label(cloud_potential, connectivity=2)
    cloud_labels_props: list = regionprops(cloud_labels, cache=True)
    cloud_labels_list: np.ndarray = np.unique(cloud_labels[cloud_labels > 0])

    ##
    # Guard if there are no clouds
    ##
    if np.size(cloud_labels_list) == 0:
        return matched_cloud_shadow_layer

    logger.debug("%s cloud labels", np.max(cloud_labels_list))

    ##
    # Label shadows
    ##
    shadow_labels: np.ndarray = label(shadow_potential, connectivity=2)

    # # Final array with matched shadows
    # matched_cloud_shadow_layer = np.zeros(cloud.shape).astype(bool)

    ##
    # Iterate labelled clouds
    ##
    for prop in cloud_labels_props:

        cloud_label_number: int = prop.label

        ##
        # Get BBox over cloud
        ##
        label_bbox = prop.bbox
        min_row: int = label_bbox[0]
        max_row: int = label_bbox[2]
        min_col: int = label_bbox[1]
        max_col: int = label_bbox[3]

        ##
        # Find exact cloud label for shadow template
        ##
        shadow_template: np.ndarray = (
            cloud_labels[min_row:max_row, min_col:max_col] == cloud_label_number
        )

        ##
        # If available, DEM elevation over label
        ##
        cloud_base_height: Union[int, float] = 0
        if dem is not None:

            # subset
            dem_subset: np.ndarray = dem[min_row:max_row, min_col:max_col]

            # get mask over and nodata
            dem_mask: np.ndarray = shadow_template & (dem_subset != -9999)
            dem_base_cloud = dem_subset[dem_mask]

            if np.size(dem_base_cloud) > 0:
                cloud_base_height = (
                    np.percentile(dem_base_cloud, 100 * high_percent) - base_dem_height
                )

        ##
        # If available, BT temp over label
        ##
        if bt is not None:

            # subset
            bt_subset: np.ndarray = bt[min_row:max_row, min_col:max_col]

            # get mask over and nodata
            bt_mask: np.ndarray = shadow_template & (bt_subset != -9999)
            bt_cloud = bt_subset[bt_mask]

            if np.size(bt_cloud) > 0:

                # area of label and effecive radius
                label_pixel_area: Union[int, float] = np.sum(shadow_template > 0)
                label_pixel_radius: Union[int, float] = np.sqrt(
                    label_pixel_area / (2 * np.pi)
                )

                cloud_base_temperature: Union[int, float] = np.min(bt_cloud)

                if label_pixel_radius >= 8:
                    per: float = (
                        100.0
                        * (label_pixel_radius - 8.0) ** 2
                        / (label_pixel_radius ** 2)
                    )
                    cloud_base_temperature = np.percentile(bt_cloud, per)

                bt_cloud[bt_cloud > cloud_base_temperature] = cloud_base_temperature

                min_cloud_height = max(
                    min_cloud_height,
                    10
                    * (temp_test_low - 400 - cloud_base_temperature)
                    / DRY_ADIABATIC_LAPSE_RATE,
                )  # in m (10 = 100deg / (deg/1000m) )

                max_cloud_height = min(
                    max_cloud_height,
                    10 * (temp_test_high + 400 - cloud_base_temperature),
                )

        ##
        # Convert cloud heights to pixels
        #
        H1_px: float = min_cloud_height / out_resolution
        H2_px: float = max_cloud_height / out_resolution

        # Range of values in px
        dx1_tmp, dy1_tmp = shadow(H1_px, sun_elevation, sun_azimuth)  # in px
        dx2_tmp, dy2_tmp = shadow(H2_px, sun_elevation, sun_azimuth)  # in px

        # Getting indices and num_steps
        longest_shift = max(abs(dx2_tmp - dx1_tmp), abs(dy2_tmp - dy1_tmp))
        num_steps = max(1, longest_shift)
        x_step = (dx2_tmp - dx1_tmp) / num_steps
        y_step = (dy2_tmp - dy1_tmp) / num_steps

        # Defining min/max values
        dx_min = min(dx1_tmp, dx2_tmp)
        dx_max = max(dx1_tmp, dx2_tmp)
        dy_min = min(dy1_tmp, dy2_tmp)
        dy_max = max(dy1_tmp, dy2_tmp)

        # Theses checking to make sure that we don't go over the size shadow_labels
        r_pad_before = 0
        r_pad_after = 0
        c_pad_before = 0
        c_pad_after = 0
        r0_shadow_ss = min_row + dy_min
        if r0_shadow_ss < 0:
            r_pad_before = abs(r0_shadow_ss)
            r0_shadow_ss = 0
        r1_shadow_ss = max_row + dy_max
        if r1_shadow_ss > shadow_labels.shape[0]:
            r_pad_after = r1_shadow_ss - shadow_labels.shape[0]
            r1_shadow_ss = shadow_labels.shape[0]
        c0_shadow_ss = min_col + dx_min
        if c0_shadow_ss < 0:
            c_pad_before = abs(c0_shadow_ss)
            c0_shadow_ss = 0
        c1_shadow_ss = max_col + dx_max
        if c1_shadow_ss > shadow_labels.shape[1]:
            c_pad_after = c1_shadow_ss - shadow_labels.shape[1]
            c1_shadow_ss = shadow_labels.shape[1]

        # we take a shadow subset to reduce memory load rather the whole image
        # this image contains subset from which patches of shadows potentials will be matched
        shadow_subset = shadow_labels[
            r0_shadow_ss:r1_shadow_ss, c0_shadow_ss:c1_shadow_ss
        ]
        shadow_subset = np.pad(
            shadow_subset,
            ((r_pad_before, r_pad_after), (c_pad_before, c_pad_after)),
            mode="constant",
            constant_values=0,
        )
        water_subset = all_water[r0_shadow_ss:r1_shadow_ss, c0_shadow_ss:c1_shadow_ss]
        water_subset = np.pad(
            water_subset,
            ((r_pad_before, r_pad_after), (c_pad_before, c_pad_after)),
            mode="constant",
            constant_values=0,
        )

        # Now, we go through heights and put similarity values into the array
        similarity_max_list = []
        index_candidate_list = []
        for i in range(0, num_steps):
            dx = int(dx1_tmp + i * x_step)
            dy = int(dy1_tmp + i * y_step)

            # we need to mask out cloud in the template
            shadow_template_tmp = np.zeros(shadow_template.shape).astype(bool)
            if (abs(dy) < shadow_template_tmp.shape[0]) & (
                abs(dx) < shadow_template_tmp.shape[1]
            ):
                # in cases when a clouds close to shadow - therefore we need to trim part of cloud
                # to match trimmed shadow
                if (dy > 0) & (dx > 0):
                    shadow_template_tmp[:-dy, :-dx] = shadow_template[dy:, dx:]
                if (dy > 0) & (dx < 0):
                    shadow_template_tmp[:-dy, -dx:] = shadow_template[dy:, :dx]
                if (dy < 0) & (dx > 0):
                    shadow_template_tmp[-dy:, :-dx] = shadow_template[:dy, dx:]
                if (dy < 0) & (dx < 0):
                    shadow_template_tmp[-dy:, -dx:] = shadow_template[:dy, :dx]

                shadow_template_tmp = np.where(
                    shadow_template_tmp & shadow_template, False, shadow_template
                )
            else:
                shadow_template_tmp = shadow_template.copy()

            # Taking shadow candidates
            # We add checkings so in cases when shadows are on the boundaries of the whole image
            r_pad_before = 0
            r_pad_after = 0
            c_pad_before = 0
            c_pad_after = 0
            r0_shadow_candidate = dy - dy_min
            if r0_shadow_candidate < 0:
                r_pad_before = abs(r0_shadow_candidate)
                r0_shadow_candidate = 0
            r1_shadow_candidate = r0_shadow_candidate + max_row - min_row
            if r1_shadow_candidate > shadow_subset.shape[0]:
                r_pad_after = r1_shadow_candidate - shadow_subset.shape[0]
                r1_shadow_candidate = shadow_subset.shape[0]
            c0_shadow_candidate = dx - dx_min
            if c0_shadow_candidate < 0:
                c_pad_before = abs(c0_shadow_candidate)
                c0_shadow_candidate = 0
            c1_shadow_candidate = c0_shadow_candidate + max_col - min_col
            if c1_shadow_candidate > shadow_subset.shape[1]:
                c_pad_after = c1_shadow_candidate - shadow_subset.shape[1]
                c1_shadow_candidate = shadow_subset.shape[1]

            shadow_candidate = shadow_subset[
                r0_shadow_candidate:r1_shadow_candidate,
                c0_shadow_candidate:c1_shadow_candidate,
            ]  # y_label[(min_row+dy):(max_row+dy), (min_col+dx):(max_col+dx)]
            shadow_candidate = np.pad(
                shadow_candidate,
                ((r_pad_before, r_pad_after), (c_pad_before, c_pad_after)),
                mode="constant",
                constant_values=0,
            )
            water_mask = water_subset[
                r0_shadow_candidate:r1_shadow_candidate,
                c0_shadow_candidate:c1_shadow_candidate,
            ]
            water_mask = np.pad(
                water_mask,
                ((r_pad_before, r_pad_after), (c_pad_before, c_pad_after)),
                mode="constant",
                constant_values=0,
            )

            # If all shadows are water
            if np.sum(shadow_candidate > 0) == np.sum(
                (shadow_candidate > 0) & (water_mask > 0)
            ):
                continue

            # similarity metric
            similarity_local_max = np.sum(
                shadow_template_tmp & (shadow_candidate > 0)
            ) / np.sum(shadow_template_tmp)

            # if it's 0, skip
            if similarity_local_max == 0:
                continue

            # getting metrics into the array
            similarity_max_list.append(similarity_local_max)
            index_candidate_list.append(i)

        # Processing similarity values
        similarity_max_arr: np.ndarray = np.array(similarity_max_list)
        index_candidate_arr: np.ndarray = np.array(index_candidate_list)

        # no matched clouds
        if np.size(similarity_max_arr) == 0:
            continue

        # This portion does the following:
        # We find first local maxima with conditions:
        # very close to the cloud baoundary
        # if it's local and small and/or continues to increase
        similarity_max = 0
        index_candidate = 0
        ind_max = 0
        for ind in range(0, similarity_max_arr.shape[0]):
            if similarity_max <= similarity_max_arr[ind]:
                similarity_max = similarity_max_arr[ind]
                index_candidate = index_candidate_arr[ind]

                ind_max = ind
            else:
                if similarity_max_arr[ind] > 0.95 * similarity_max:
                    continue
                else:
                    # if less than 0.3
                    if similarity_max < 0.3:
                        continue

                    # avoiding a situation when max is reached over the very boundary of clouds
                    dx_cloud = int(dx1_tmp + index_candidate_arr[ind_max] * x_step)
                    dy_cloud = int(dy1_tmp + index_candidate_arr[ind_max] * y_step)

                    # if near to the cloud
                    if (
                        np.sqrt(dx_cloud * dx_cloud + dy_cloud * dy_cloud)
                        <= neighbour_tolerance
                    ):  # sqrt(3^2+3^2)
                        continue

                    break

        # take subset for the first minimum + 0.95*max
        similarity_max_arr = similarity_max_arr[0 : (ind + 1)]
        index_candidate_arr = index_candidate_arr[0 : (ind + 1)]
        mm = similarity_max_arr >= similarity_matched_threshold * similarity_max
        index_candidate_arr = index_candidate_arr[mm]
        dx_matched_arr = (dx1_tmp + index_candidate_arr * x_step).astype(int)
        dy_matched_arr = (dy1_tmp + index_candidate_arr * y_step).astype(int)

        if similarity_max > 0.3:
            for index_candidate in index_candidate_arr:
                dx_matched = int(dx1_tmp + index_candidate * x_step)
                dy_matched = int(dy1_tmp + index_candidate * y_step)

                r0_shift = 0
                r1_shift = shadow_template.shape[0]
                c0_shift = 0
                c1_shift = shadow_template.shape[1]

                r0_final_shadow = min_row + dy_matched
                if r0_final_shadow < 0:
                    r0_shift = abs(r0_final_shadow)
                    r0_final_shadow = 0

                r1_final_shadow = max_row + dy_matched
                if r1_final_shadow > matched_cloud_shadow_layer.shape[0]:
                    r1_shift = r1_shift - (
                        r1_final_shadow - matched_cloud_shadow_layer.shape[0]
                    )
                    r1_final_shadow = matched_cloud_shadow_layer.shape[0]

                c0_final_shadow = min_col + dx_matched
                if c0_final_shadow < 0:
                    c0_shift = abs(c0_final_shadow)
                    c0_final_shadow = 0

                c1_final_shadow = max_col + dx_matched
                if c1_final_shadow > matched_cloud_shadow_layer.shape[1]:
                    c1_shift = c1_shift - (
                        c1_final_shadow - matched_cloud_shadow_layer.shape[1]
                    )
                    c1_final_shadow = matched_cloud_shadow_layer.shape[1]

                matched_cloud_shadow_layer[
                    r0_final_shadow:r1_final_shadow, c0_final_shadow:c1_final_shadow
                ] = (
                    matched_cloud_shadow_layer[
                        r0_final_shadow:r1_final_shadow,
                        c0_final_shadow:c1_final_shadow,
                    ]
                    | shadow_template[r0_shift:r1_shift, c0_shift:c1_shift]
                )

    logger.debug(
        "Sum of matched cloud shadow layer %s", np.sum(matched_cloud_shadow_layer)
    )
    return matched_cloud_shadow_layer
