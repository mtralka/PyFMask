from enum import Enum
from pathlib import Path
from typing import Dict
from typing import Union

from pyfmask.classes import SensorData


class PlatformBase:
    class Bands(Enum):
        ...

    RGB: tuple
    RESAMPLE_BANDS: tuple

    CLOUD_THRESHOLD: float
    PROBABILITY_WEIGHT: float  # for thin/cirrus clouds
    OUT_RESOLUTION: int
    NO_DATA: int

    @staticmethod
    def calculate_erosion_pixels(out_resolution: int) -> int:
        return round(90.0 / out_resolution)

    @staticmethod
    def is_platform(file_path: Union[Path, str]) -> bool:
        """Determines if given `file_path` is of class platform type"""
        raise NotImplementedError

    @classmethod
    def _get_calibration_parameters(cls, file_path: Path) -> dict:
        """Returns extracted calibration parameters from file at `file_path`"""
        raise NotImplementedError

    @classmethod
    def _get_file_names(cls, file_path: Path) -> Dict[str, Path]:
        """Return Dict of band names and corresponding path queried from `file_path`"""
        raise NotImplementedError

    @classmethod
    def get_data(cls, file_path: Union[Path, str]) -> SensorData:
        """Returns `SensorData` type of platform info from `file_path`"""
        raise NotImplementedError
