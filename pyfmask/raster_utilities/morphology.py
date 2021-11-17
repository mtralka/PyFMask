import numpy as np
from scipy.ndimage import uniform_filter


def focal_variance(image_arr: np.ndarray, window_size: int = 7) -> np.ndarray:
    
    img32: np.ndarray = image_arr.astype(np.float32)
    focal_mean = uniform_filter(img32, size=window_size, mode='constant', cval=0)
    mean_square = uniform_filter(img32**2, size=window_size, mode='constant', cval=0)
    variance = mean_square - focal_mean * focal_mean

    return variance
