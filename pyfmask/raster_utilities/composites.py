import numpy as np
from pyfmask.raster_utilities.morphology import focal_variance


def create_ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return (nir - red) / (nir + red + eps)


def create_ndsi(green: np.ndarray, swir1: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return (green - swir1) / (green + swir1 + eps)


def create_ndbi(swir1: np.ndarray, nir: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return (swir1 - nir) / (swir1 + nir + eps)


def create_cdi(
    nir: np.ndarray,
    nir2: np.ndarray,
    red3: np.ndarray,
    window_size: int = 7,
    eps: float = 1e-7,
) -> np.ndarray:

    ratio_b8_b8a: np.ndarray = nir / (nir2 + eps)
    ratio_b7_b8a: np.ndarray = red3 / (nir2 + eps)

    variance_b8_b8a = focal_variance(ratio_b8_b8a, window_size=window_size)
    variance_b7_b8a = focal_variance(ratio_b7_b8a, window_size=window_size)

    cdi = np.zeros(variance_b8_b8a.shape, dtype=np.float32)

    mask_non_zero = (variance_b7_b8a + variance_b8_b8a) != 0

    cdi[mask_non_zero] = (
        variance_b7_b8a[mask_non_zero] - variance_b8_b8a[mask_non_zero]
    ) / (variance_b7_b8a[mask_non_zero] + variance_b8_b8a[mask_non_zero])

    return cdi
