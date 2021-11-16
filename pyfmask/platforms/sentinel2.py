from dataclasses import dataclass
from enum import Enum
from enum import auto
from os import stat
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import gdal
import numpy as np
from pyfmask.utils.classes import SensorData

# from pyfmask.utils.classes import SupportedSensors
from pyfmask.utils.metadata import extract_metadata
from pyfmask.utils.raster_utils import NO_DATA


class Sentinel2:
    class Bands(Enum):
        BLUE = 2
        GREEN = 3
        RED = 4
        RED3 = 7
        NIR = 8
        NIR2 = "8A"
        SWIR1 = 11
        SWIR2 = 12
        CIRRUS = 10

    RGB: tuple = (Bands.RED, Bands.GREEN, Bands.BLUE)

    @staticmethod
    def is_platform(file_path: Union[Path, str]) -> bool:
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        file_name = file_path.name

        if "'MTD_TL.xml" in file_name:
            return True
        return False

    @classmethod
    def _get_calibration_parameters(cls, file_path: Path) -> dict:

        target_attributes: list = ["AZIMUTH_ANGLE", "ZENITH_ANGLE"]

        metadata: dict = extract_metadata(file_path, target_attributes)

        return metadata

    @classmethod
    def _get_file_names(cls, file_path: Path) -> dict:

        attributes: list = ["FILE_NAME_BAND_{}"]
        target_attributes: list = []

        for target in attributes:
            for band in cls.Bands.__members__.values():
                target_attributes.append(target.format(band.value))

        file_names: dict = extract_metadata(file_path, target_attributes)

        band_names: List[cls.Bands] = [b.name for b in cls.Bands]
        return {k: v for k, v in zip(band_names, file_names.values())}

    @classmethod
    def get_data(cls, file_path: Union[Path, str]) -> SensorData:

        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        parameters = SensorData()

        calibration = cls._get_calibration_parameters(file_path)
        file_band_names = cls._get_file_names(file_path)

        parameters.sensor = "S2_MSI"  # SupportedSensors.L08_OLI
        parameters.sun_elevation = 90.0 - calibration["ZENITH_ANGLE"]
        parameters.sun_azimuth = calibration["AZIMUTH_ANGLE"]
        parameters.scene_id = file_path.name  # TODO redo this

        for band in cls.Bands.__members__.values():

            band_number = band.value
            band_name = band.name

            band_ds = None
            band_array = None

            # band_ds = gdal.Open(file_band_names[band_name])
            # band_array = band_ds.GetRasterBand(1).ReadAsArray().astype(np.unint16)

            ##
            # Upsample to 20m
            ##
            if band in cls.RGB:
                # GDAL WARP
                # OPEN DS
                ...

            elif band == cls.Bands.CIRRUS:
                # GDAL WARP
                # OPEN DS
                ...

            else:
                # CRETE DS
                # OPEN DS
                ...

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
            # processed_band_array: np.ndarray

            # if band != cls.Bands.BT:
            #     processed_band_array = band_array * parameters.calibration[band_name]
            #     processed_band_array = (
            #         1000
            #         * processed_band_array
            #         / np.sin(calibration["SUN_ELEVATION"] * np.pi / 180)
            #     )

            # elif band == cls.Bands.BT:

            #     # convert to TOA
            #     toa_array: np.ndarray = (
            #         band_array * calibration[f"REFLECTANCE_MULT_BAND_{band_number}"]
            #         + calibration[f"REFLECTANCE_ADD_BAND_{band_number}"]
            #     )

            #     # convert to kelvin
            #     kelvin_array: np.ndarray = (
            #         calibration[f"K2_CONSTANT_BAND_{band_number}"]
            #     ) / np.log(
            #         calibration[f"K1_CONSTANT_BAND_{band_number}"] / toa_array + 1
            #     )

            #     # convert to celsisus and scale
            #     celsius_array: np.ndarray = 100 * (kelvin_array - 273.15)

            #     processed_band_array = celsius_array

            ##
            # Assign NoData
            ##
            processed_band_array = np.where(
                band_array == 0, NO_DATA, band_array
            ).astype(np.int16)

            ##
            # Assign band into object
            ##
            setattr(parameters, band_name, processed_band_array)

            return parameters
