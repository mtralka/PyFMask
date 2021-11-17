from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import gdal
import numpy as np

# from pyfmask.utils.classes import SupportedSensors
from pyfmask.extractors.metadata import extract_metadata
from pyfmask.platforms.platform_utils import calculate_erosion_pixels
from pyfmask.utils.classes import SensorData
from pyfmask.utils.raster_utils import NO_DATA


class Landsat8:
    class Bands(Enum):
        BLUE = 2
        GREEN = 3
        RED = 4
        NIR = 5
        SWIR1 = 6
        SWIR2 = 7
        CIRRUS = 9
        BT = 10

    RGB: tuple = (Bands.RED, Bands.GREEN, Bands.BLUE)

    @staticmethod
    def is_platform(file_path: Union[Path, str]) -> bool:
        print(file_path)
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        file_name = file_path.name

        if ("LC08" in file_name) & ("_MTL.txt" in file_name):
            return True
        return False

    @classmethod
    def _get_calibration_parameters(cls, file_path: Path) -> dict:

        attributes: list = ["REFLECTANCE_MULT_BAND_{}", "REFLECTANCE_ADD_BAND_{}"]
        target_attributes: list = [
            "SUN_ELEVATION",
            "K1_CONSTANT_BAND_10",
            "K2_CONSTANT_BAND_10",
            "RADIANCE_ADD_BAND_10",
            "RADIANCE_MULT_BAND_10",
            "SUN_AZIMUTH",
        ]

        for target in attributes:
            for band in cls.Bands.__members__.values():
                if band == cls.Bands.BT:
                    continue
                target_attributes.append(target.format(band.value))

        landsat_metadata: Dict[str, str] = extract_metadata(
            file_path, target_attributes
        )

        return {k: float(v) for k, v in landsat_metadata.items()}

    @classmethod
    def _get_file_names(cls, file_path: Path) -> dict:

        attributes: list = ["FILE_NAME_BAND_{}"]
        target_attributes: list = []

        for target in attributes:
            for band in cls.Bands.__members__.values():
                target_attributes.append(target.format(band.value))

        file_names: dict = extract_metadata(file_path, target_attributes)

        band_names: List[cls.Bands] = [b.name for b in cls.Bands]

        return {k: v.strip('"') for k, v in zip(band_names, file_names.values())}

    @classmethod
    def get_data(cls, file_path: Union[Path, str]) -> SensorData:

        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        parameters = SensorData(
            cloud_threshold=17.5, probability_weight=0.3, out_resolution=30
        )

        calibration = cls._get_calibration_parameters(file_path)
        file_band_names = cls._get_file_names(file_path)

        parameters.file_band_names = file_band_names
        parameters.sensor = "L08_OLI"  # SupportedSensors.L08_OLI
        parameters.sun_azimuth = float(calibration.pop("SUN_AZIMUTH"))
        parameters.sun_elevation = float(calibration.pop("SUN_ELEVATION"))
        parameters.calibration = calibration
        parameters.scene_id = file_path.name.split("_MTL.txt")[0]  # TODO REDO this
        parameters.erode_pixels = calculate_erosion_pixels(parameters.out_resolution)

        parameters.band_data = {}

        for band in cls.Bands.__members__.values():
            print(band)
            band_number = band.value
            band_name = band.name

            band_path: Path = file_path.parent / file_band_names[band_name]

            band_ds = gdal.Open(str(band_path))
            band_array = band_ds.GetRasterBand(1).ReadAsArray().astype(np.uintc)

            ##
            # Use RED band as projection base
            ##
            if band == cls.Bands.RED:
                parameters.geo_transform = band_ds.GetGeoTransform()
                parameters.projection_reference = band_ds.GetProjectionRef()

            ##
            # NoData
            ##
            if not hasattr(parameters, "nodata_mask"):
                parameters.nodata_mask = band_array == 0
            else:
                parameters.nodata_mask = (parameters == True) | (band_array == 0)

            ##
            # Saturation of visible bands (RGB)
            ##
            if not hasattr(parameters, "vis_saturation"):
                parameters.vis_saturation = np.zero(band_array.shape).astype(np.bool)

            if band in cls.RGB:
                parameters.vis_saturation = np.where(
                    band_array == 65535, True, parameters.vis_saturation
                )

            ##
            # Convert to TOA reflectance
            ##
            processed_band_array: np.ndarray

            if band != cls.Bands.BT:
                processed_band_array = (
                    band_array * calibration[f"REFLECTANCE_MULT_BAND_{band_number}"]
                    + calibration[f"REFLECTANCE_ADD_BAND_{band_number}"]
                )
                processed_band_array = (
                    1000
                    * processed_band_array
                    / np.sin(parameters.sun_elevation * np.pi / 180)
                )

            elif band == cls.Bands.BT:

                # convert to TOA
                toa_array: np.ndarray = (
                    band_array * calibration[f"RADIANCE_MULT_BAND_{band_number}"]
                    + calibration[f"RADIANCE_ADD_BAND_{band_number}"]
                )

                # convert to kelvin
                kelvin_array: np.ndarray = (
                    calibration[f"K2_CONSTANT_BAND_{band_number}"]
                ) / np.log(
                    calibration[f"K1_CONSTANT_BAND_{band_number}"] / toa_array + 1
                )

                # convert to celsisus and scale
                celsius_array: np.ndarray = 100 * (kelvin_array - 273.15)

                processed_band_array = celsius_array

            processed_band_array = np.where(
                band_array == 0, NO_DATA, processed_band_array
            ).astype(np.int16)

            parameters.band_data[band_name] = processed_band_array

        parameters.x_size = parameters.band_data["RED"].shape[1]
        parameters.y_size = parameters.band_data["RED"].shape[0]

        return parameters
