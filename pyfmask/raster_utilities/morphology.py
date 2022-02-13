from typing import List
from typing import Optional
from typing import Union

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter
from skimage import morphology
from skimage.measure import label


def dilate_array(array: np.ndarray, amount: Union[float, int]) -> np.ndarray:
    return morphology.binary_dilation(
        array.astype(bool), morphology.square(2 * amount + 1)
    )


def focal_variance(array: np.ndarray, window_size: int = 7) -> np.ndarray:

    img32: np.ndarray = array.astype(np.float32)
    focal_mean = uniform_filter(img32, size=window_size, mode="constant", cval=0)
    mean_square = uniform_filter(img32 ** 2, size=window_size, mode="constant", cval=0)
    variance = mean_square - focal_mean * focal_mean

    return variance


def enhance_line(array: np.ndarray) -> np.ndarray:
    """Enhance line array"""

    template: List[np.ndarray] = []
    template.append(np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]) / 6.0)

    template.append(np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]) / 6.0)

    template.append(np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]) / 6.0)

    template.append(np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]) / 6.0)

    array_result: np.ndarray = -9999999 * np.ones(array.shape).astype(np.float32)

    for k in template:
        tmp = signal.convolve2d(array, k, mode="same", boundary="fill", fillvalue=0)
        array_result = np.maximum(array_result, tmp)

    return array_result


def erode_commissons(
    cdi: Optional[np.ndarray],
    potential_clouds: np.ndarray,
    potential_false_positives: np.ndarray,
    water: np.ndarray,
    erode_pixels: int,
) -> np.ndarray:
    """Erode pixels representing cloud comission errors"""

    cloud: np.ndarray = np.where(potential_clouds > 0, True, False)

    ##
    # Erode potential false cloud pixels
    ##
    clouds_after_erosion: np.ndarray = morphology.binary_erosion(
        cloud, morphology.disk(erode_pixels)
    )
    pixels_eroded: np.ndarray = (clouds_after_erosion == False) & (
        potential_false_positives == True
    )

    ##
    # Remove `potential_false_positive` pixels
    ##
    clouds_after_erosion = np.array(cloud, copy=True)  # must deep copy
    clouds_after_erosion = np.where(pixels_eroded == True, 0, clouds_after_erosion)

    ##
    # Dilate to orginal cloud shape
    ##
    clouds_redilated: np.ndarray = morphology.binary_dilation(
        clouds_after_erosion, morphology.disk(2 * erode_pixels)
    )

    # Remover the clouds gone forever
    # Segmentate each cloud to remove the small objs
    cloud_labels: np.ndarray = label(
        cloud, connectivity=2
    )  # corresponds to connectivity 8
    clouds_remaining = np.array(cloud_labels, copy=True)
    clouds_remaining = np.where(
        clouds_after_erosion == False, 0, clouds_remaining
    )  # remove the clouds gone.
    idx_clouds_remaining = np.unique(clouds_remaining)

    cloud_remaining = np.zeros(
        cloud.shape, dtype=bool
    )  # the clouds needed to be eroded.
    if np.size(idx_clouds_remaining) > 0:
        idx_clouds_remaining = np.delete(
            idx_clouds_remaining, np.where(idx_clouds_remaining == 0)
        )

        if np.size(idx_clouds_remaining) > 0:
            cloud_remaining = np.isin(cloud_labels, idx_clouds_remaining)

    # only for land
    cloud = ((clouds_redilated == True) & (cloud_remaining == True)) | (
        (water == True) & (cloud == True)
    )  # add clouds over water

    if cdi is None:
        return cloud  # if not Sentinel-2

    ##
    # Remove small object with CDI < -0.5, only for Sentinel 2
    ##
    large_objects: np.ndarray = morphology.remove_small_objects(
        cloud.astype(bool), 10000, connectivity=2
    )

    # invert `large_objects` to obtain small objects
    small_objects: np.ndarray = (cloud == 1) & (large_objects == 0)

    # label small clouds
    small_object_labels: np.ndarray = label(small_objects, connectivity=2)

    # identify all cloud pixels below CDI threshold
    confident_cloud_pixels: np.ndarray = cdi < -0.5

    # remove non-confident cloud pixels
    small_objects_excluding_urban: np.ndarray = np.array(small_object_labels, copy=True)
    small_objects_excluding_urban[confident_cloud_pixels == 0] = 0

    ##
    # Identify `true_clouds` as areas where `confident_cloud_pixels` remain
    idx = np.unique(small_objects_excluding_urban)
    true_cloud = np.isin(small_object_labels, idx)

    ##
    # Remove bright surfaces
    ##
    bright_surfaces = (true_cloud == 0) & (small_objects == 1)
    cloud[bright_surfaces] = 0

    ##
    # Remove very small objects
    ##
    cloud = morphology.remove_small_objects(cloud.astype(bool), 3, connectivity=2)

    return cloud
