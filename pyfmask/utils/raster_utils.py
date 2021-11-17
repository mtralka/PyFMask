import numpy as np

NO_DATA = -9999


def create_ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return (nir - red) / (nir + red + eps)


def create_ndsi(green: np.ndarray, swir1: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return (green - swir1) / (green + swir1 + eps)


def create_ndbi(swir1: np.ndarray, nir: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return (swir1 - nir) / (swir1 + nir + eps)
