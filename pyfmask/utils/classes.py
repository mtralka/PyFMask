from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np


# from pyfmask.platforms.landsat8 import Landsat8
# from pyfmask.platforms.sentinel2 import Sentinel2


# class SupportedSensors(Enum):
#     L08_OLI = Landsat8
#     S2_MSI = Sentinel2

# class SupportedSensors(Enum):
# #     L08_OLI = Landsat8
# #     S2_MSI = Sentinel2


@dataclass
class SensorData:
    cloud_threshold: float
    probability_weight: float
    out_resolution: int
    x_size: int
    y_size: int
    erode_pixels: int
    sensor: str
    scene_id: str
    sun_elevation: Union[float, int]
    sun_azimuth: Union[float, int]
    geo_transform: tuple
    projection_reference: tuple
    calibration: Any
    file_band_names: List[str]
    nodata_mask: np.ndarray
    vis_saturation: np.ndarray
    band_data: Dict[str, np.ndarray]


@dataclass
class DEMData:
    dem: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray


@dataclass
class GSWOData:
    gswo: np.ndarray


@dataclass
class PotentialCloudPixels:
    potential_pixels: np.ndarray
    whiteness: np.ndarray
    hot: np.ndarray
    normalized_cirrus: Optional[np.ndarray]
