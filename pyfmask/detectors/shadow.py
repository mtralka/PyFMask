import numpy as np
from skimage import morphology


def detect_potential_cloud_shadow(data, data_clear_land):
    # data: data dictionary
    # data_clear_land: clear pixels (not potential clouds) over land and w/o nodata
    percent_low = 0.175
    sun_zenith_deg = 90.0 - data["sun_elev"]
    sun_azimuth_deg = data["sun_azimuth"]

    if ("slope" in data.keys()) & ("aspect" in data.keys()):
        # to perform topo correction
        data_nir, data_swir1 = get_data_topo_corrected(
            data, sun_zenith_deg, sun_azimuth_deg
        )
    else:
        data_nir = data["nir"]
        data_swir1 = data["swir1"]

    # NIR/SWIR flood fill
    backg_b4 = np.percentile(data["nir"][data_clear_land], 100 * percent_low)
    backg_b5 = np.percentile(data["swir1"][data_clear_land], 100 * percent_low)

    data_nir = np.where(
        (data["nodata_mask"]) | (np.isnan(data_nir)), backg_b4, data_nir
    )
    data_nir_filled = imfill_skimage(data_nir.astype(np.float32))
    data_nir_dif = data_nir_filled - data_nir

    data_swir1 = np.where(
        (data["nodata_mask"]) | (np.isnan(data_swir1)), backg_b5, data_swir1
    )
    data_swir1_filled = imfill_skimage(data_swir1.astype(np.float32))
    data_swir1_dif = data_swir1_filled - data_swir1

    # compute shadow probability
    shadow_prob = np.minimum(data_nir_dif, data_swir1_dif)

    # computing potential shadow layer with threshold 200 [Original threhsold]
    # However, probably there are differences in imfill implementation [matlab and python]
    # It results in many shadow over detection (too conservative)
    # We increase it to 500
    shadow_mask = np.where(shadow_prob > 500, 1, 0).astype(np.uint8)
    # we remove potential shadows of 3 pixels as with clouds
    shadow_mask = morphology.remove_small_objects(
        shadow_mask.astype(bool), 3, connectivity=2
    )
    shadow_mask = np.where(data["nodata_mask"], 255, shadow_mask)

    return shadow_mask


def get_data_topo_corrected(data, sun_zenith_deg, sun_azimuth_deg):
    # TBD: provide topo correction for NIR and SWIR
    data_nir = data["nir"]
    data_swir1 = data["swir1"]

    return data_nir, data_swir1


def imfill_skimage(img):
    """
    Replicates the imfill function available within MATLAB. Based on the
    example provided in
    https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html
    """
    seed = img.copy()
    # Define seed points and the start points for the erosion process.
    seed[1:-1, 1:-1] = img.max()
    # Define the mask; Probably unneeded.
    mask = img
    # Fill the holes
    filled = morphology.reconstruction(seed, mask, method="erosion")

    return filled
