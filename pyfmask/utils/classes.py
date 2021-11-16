from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np


# from pyfmask.platforms.landsat8 import Landsat8
# from pyfmask.platforms.sentinel2 import Sentinel2


# class SupportedSensors(Enum):
#     L08_OLI = Landsat8
#     S2_MSI = Sentinel2


@dataclass
class SensorData:
    cloud_threshold: float
    probability_weight: float
    out_resolution: int
    x_size: Optional[int] = None
    y_size: Optional[int] = None
    erode_pixels: Optional[int] = None
    sensor: Optional[Any] = None
    scene_id: Optional[int] = None
    sun_elevation: Optional[Union[float, int]] = None
    sun_azimuth: Optional[Union[float, int]] = None
    geo_transform: Optional[tuple] = None  # CHECK
    projection_reference: Optional[tuple] = None  # CHECK
    calibration: Optional[Any] = None
    file_band_names: Optional[Any] = None
    nodata_mask: Optional[Any] = None
    vis_saturation: Optional[np.ndarray] = None
    band_data: Optional[Dict[str, np.ndarray]] = None
