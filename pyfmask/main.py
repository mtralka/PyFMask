from enum import auto
from logging.config import valid_ident
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from typing import ValuesView
from typing import cast

import gdalconst
import numpy as np

from pyfmask.detectors import detect_absolute_snow
from pyfmask.detectors import detect_snow
from pyfmask.detectors import detect_water
from pyfmask.detectors.cloud import detect_false_positive_cloud_pixels
from pyfmask.detectors.cloud import detect_potential_clouds
from pyfmask.detectors.cloud import detect_potential_cloud_pixels
from pyfmask.extractors.auxillary_data import AuxTypes
from pyfmask.extractors.auxillary_data import extract_aux_data
from pyfmask.platforms.landsat8 import Landsat8
from pyfmask.raster_utilities.composites import create_cdi
from pyfmask.raster_utilities.composites import create_ndbi
from pyfmask.raster_utilities.composites import create_ndsi
from pyfmask.raster_utilities.composites import create_ndvi
from pyfmask.raster_utilities.io import create_outfile_dataset
from pyfmask.raster_utilities.io import write_array_to_ds
from pyfmask.raster_utilities.morphology import dilate_array
from pyfmask.raster_utilities.morphology import enhance_line
from pyfmask.raster_utilities.morphology import erode_commissons
from pyfmask.utils.classes import DEMData
from pyfmask.utils.classes import GSWOData
from pyfmask.utils.classes import PotentialCloudPixels
from pyfmask.utils.classes import PotentialClouds
from pyfmask.utils.classes import SensorData
from pyfmask.utils.utils import valdiate_path


class fmask:
    def __init__(
        self,
        infile: Union[Path, str],
        out_dir: Union[Path, str],
        out_name: str,
        dem_path: Union[Path, str],
        gswo_path: Union[Path, str],
        cloud_threshold: Optional[Union[float, int]] = None,
        dilated_cloud_px: int = 3,
        dilated_shadow_px: int = 3,
        dilated_snow_px: int = 0,
        output_cloud_prob: bool = True,
        dem_nodata: Union[float, int] = -9999,
        gswo_nodata: Union[float, int] = 255,
        auto_save: bool = True,
    ):
        self.infile: Path = valdiate_path(infile, check_exists=True, check_is_file=True)
        self.out_dir: Path = valdiate_path(out_dir, check_is_dir=True)
        self.out_name: str = out_name
        self.dem_path: Path = valdiate_path(
            dem_path, check_exists=True, check_is_dir=True
        )
        self.gswo_path: Path = valdiate_path(
            gswo_path, check_exists=True, check_is_dir=True
        )

        self.auto_save: bool = auto_save

        if cloud_threshold:
            self.cloud_threshold: float = float(cloud_threshold)

        self.dilated_cloud_px: int = dilated_cloud_px
        self.dilated_shadow_px: int = dilated_shadow_px
        self.dilated_snow_px: int = dilated_snow_px

        self.output_cloud_prob: bool = output_cloud_prob

        self.dem_nodata: Union[float, int] = dem_nodata
        self.gswo_nodata: Union[float, int] = gswo_nodata

        self.temp_dir: Path

        self.platform_data: SensorData
        self.dem_data: Optional[DEMData]
        self.gswo_data: Optional[GSWOData]

        self.ndvi: np.ndarray
        self.ndsi: np.ndarray
        self.ndbi: np.ndarray

        self.potential_cloud_pixels: PotentialCloudPixels
        self.potential_clouds: PotentialClouds

        self.cdi: Optional[np.ndarray] = None
        self.absolute_snow: np.ndarray

        self.snow: np.ndarray
        self.water: np.ndarray
        self.cloud_shadow: np.ndarray
        self.cloud: np.ndarray

        self.results: np.ndarray

    def run(self) -> None:

        self._extract_platform_data()

        self.temp_dir = self._create_temp_directory()

        self._extract_dem_data()

        self._create_spectral_composites()

        ##
        # Detect snow pixels
        ##
        self.snow = detect_snow(
            self.ndsi,
            band_data=self.platform_data.band_data,
        )

        ##
        # Detect water pixels
        ##
        self.water = detect_water(
            self.platform_data.band_data["NIR"],
            self.ndvi,
            self.platform_data.nodata_mask,
            self.snow,
            cast(GSWOData, self.gswo_data),
        )  # TODO adjust this

        ##
        # Calculate CDI if platform is S2
        ##
        # TODO test this
        if self.platform_data.sensor == "S2_MSI":
            self.cdi = create_cdi(
                self.platform_data.band_data["NIR"],
                self.platform_data.band_data["NIR2"],
                self.platform_data.band_data["RED3"],
            )

        ##
        # Detect potential cloud pixels
        ##
        self.potential_cloud_pixels = detect_potential_cloud_pixels(
            ndsi=self.ndsi,
            ndvi=self.ndvi,
            band_data=self.platform_data.band_data,
            vis_saturation=self.platform_data.vis_saturation,
            dem_data=self.dem_data,
            nodata_mask=self.platform_data.nodata_mask,
        )

        print("probable clouds", np.sum(self.potential_cloud_pixels.potential_pixels))

        ##
        # Update cirrus clouds
        ##
        if self.platform_data.band_data.get("CIRRUS", None) is not None:
            self.platform_data.band_data["CIRRUS"] = cast(
                np.ndarray, self.potential_cloud_pixels.normalized_cirrus
            )

        ##
        # Detect absolute snow (pure snow / ice)
        ##
        self.absolute_snow = detect_absolute_snow(
            self.platform_data.sensor,
            self.snow,
            self.platform_data.band_data,
            self.platform_data.vis_saturation,
            self.ndsi,
        )

        ##
        # Run cloud detection processes
        ##
        self._run_cloud_computations()

        ##
        # Cloud shadow detection
        ##
        self._run_cloud_shadow_computations()

        ##
        # Dilate snow, shadows, and clouds
        ##
        if self.dilated_snow_px > 0:
            self.snow = dilate_array(self.snow, self.dilated_snow_px)
        # if self.dilated_shadow_px > 0:
        # self.cloud_shadow = dilate_array(self.cloud_shadow, self.dilated_shadow_px)
        if self.dilated_cloud_px > 0:
            self.cloud = dilate_array(self.cloud, self.dilated_cloud_px)

        ##
        # Compute final results
        ##
        self._compute_final_results()

        if self.auto_save:
            self.save_results()

    def _extract_platform_data(self) -> None:
        """Extract platform data"""
        supported_platforms: Dict[str, Any] = {"Landsat8": Landsat8}

        for name, platform_object in supported_platforms.items():
            if platform_object.is_platform(self.infile):
                print(f"Identified as {name}")
                self.platform_data = platform_object.get_data(
                    self.infile
                )  # TODO something withh threshold class attrs
                break
        else:
            raise ValueError("Platform not found or supported")

    def _extract_dem_data(self) -> None:
        """Extract data from DEMs and GSWOs"""

        aux_data_kwargs: Dict[str, Any] = {
            "projection_reference": self.platform_data.projection_reference,
            "x_size": self.platform_data.x_size,
            "y_size": self.platform_data.y_size,
            "geo_transform": self.platform_data.geo_transform,
            "out_resolution": self.platform_data.out_resolution,
            "scene_id": self.platform_data.scene_id,
            "temp_dir": self.temp_dir,
        }

        dem_data = extract_aux_data(
            aux_path=self.dem_path,
            aux_type=AuxTypes.DEM,
            no_data=self.dem_nodata,
            **aux_data_kwargs,
        )

        self.dem_data = cast(DEMData, dem_data) if dem_data else None

        gswo_data = extract_aux_data(
            aux_path=self.gswo_path,
            aux_type=AuxTypes.GSWO,
            no_data=self.gswo_nodata,
            **aux_data_kwargs,
        )

        self.gswo_data = cast(GSWOData, gswo_data) if gswo_data else None

    def _create_spectral_composites(self) -> None:

        self.ndvi = create_ndvi(
            self.platform_data.band_data["RED"], self.platform_data.band_data["NIR"]
        )
        self.ndsi = create_ndsi(
            self.platform_data.band_data["GREEN"], self.platform_data.band_data["SWIR1"]
        )
        self.ndbi = create_ndbi(
            self.platform_data.band_data["SWIR1"], self.platform_data.band_data["NIR"]
        )

    def _create_temp_directory(self) -> Path:

        temp_name: str = str(self.platform_data.scene_id) + "_temp"
        outfile_path: Path = self.out_dir / temp_name

        if outfile_path.exists():
            print("WARN, outfile path exists, rewriting")

        outfile_path.mkdir(exist_ok=True)

        return outfile_path

    def _run_cloud_computations(self) -> None:
        """Run cloud detection pipeline"""

        ##
        # Detect probable clouds
        ##
        self.potential_clouds = detect_potential_clouds(
            self.platform_data.band_data,
            self.dem_data,
            self.potential_cloud_pixels,
            self.platform_data.nodata_mask,
            self.water,
            self.platform_data.probability_weight,  # thin / cirrus weight
            self.platform_data.cloud_threshold,  # cloud prob threshold
            self.ndsi,
            self.ndvi,
            self.ndbi,
            self.platform_data.vis_saturation,
        )

        # logged
        # t_templ, t_temph

        ##
        # Update DEM normalized BT clouds
        ##
        if (
            self.platform_data.band_data.get("BT", None) is not None
            and self.potential_clouds.bt_normalized_dem is not None
        ):
            self.platform_data.band_data["BT"] = cast(
                np.ndarray, self.potential_clouds.bt_normalized_dem
            )

        ##
        # Enhance NDBI for high urban/built-up areas
        ##
        self.ndbi = enhance_line(self.ndbi)

        ##
        # Detect potential false positives in cloud layer
        ##
        potential_false_positives: np.ndarray = (
            detect_false_positive_cloud_pixels(
                self.platform_data.band_data,
                self.ndbi,
                self.ndvi,
                self.platform_data,
                self.snow,
                self.water,
                self.potential_clouds.cloud,
                self.cdi,
                self.dem_data,
            )
        )

        ##
        # Remove commission errors from urban, bright rock, coastline
        ##
        self.cloud = erode_commissons(
            self.cdi,
            self.potential_clouds.cloud,
            potential_false_positives,
            self.water,
            self.platform_data.erode_pixels,
        )

    def _run_cloud_shadow_computations(self) -> None:

        # shadow_matched = np.zeros(cs_final.shape, np.uint8)

        #     # Shadow detection
        # if (sum_clr <= 40000):
        #     print(f'No clear pixel in this image (clear-sky pixels = {sum_clr})')
        #     pcloud = (pcloud>0)
        #     pshadow = np.where(pcloud==0,1,0)
        #     shadow_matched = pshadow
        # else:
        #     print('Match cloud shadows with clouds')
        #     # detect potential cloud shadow
        #     t0 = time.time()

        #     # Performing potential shadows search
        #     pshadow = detect_potential_cloud_shadow(data, idlnd)
        #     # Performaing shadows matching
        #     shadow_matched = match_cloud_shadow(data, pcloud, pshadow, water_all, sum_clr, t_templ, t_temph)

        #     print(f'Processing time of shadow detection: {time.time()-t0} sec')
        ...

    def _compute_final_results(self) -> None:

        final_array: np.ndarray = np.zeros(
            self.potential_clouds.cloud.shape, dtype=np.uint8
        )

        ##
        # Water
        ##
        # final_array = np.where(self.water==1, 1, final_array) TODO
        final_array = np.where(self.water > 0, 1, final_array)

        ##
        # Snow
        ##
        final_array = np.where(self.snow > 0, 3, final_array)

        ##
        # Shadow
        ##
        # final_array = np.where(shadow_matched > 0, 2, final_array)

        ##
        # Cloud
        ##
        final_array = np.where(self.cloud > 0, 4, final_array)

        ##
        # No Data
        ##
        final_array = np.where(self.platform_data.nodata_mask, 255, final_array)

        self.results = final_array

    def save_results(self) -> None:

        ##
        # Make output folder
        ##

        x_size: int = self.potential_clouds.cloud.shape[1]
        y_size: int = self.potential_clouds.cloud.shape[0]
        outfile_path: str = str(self.out_dir / self.out_name)
        ##
        # Create output dataset
        ##
        outfile_ds = create_outfile_dataset(
            outfile_path,
            x_size,
            y_size,
            self.platform_data.projection_reference,
            self.platform_data.geo_transform,
            1,
            data_type=gdalconst.GDT_Byte,
        )
        outfile_ds = write_array_to_ds(outfile_ds, self.results)
        outfile_ds = None
        # if save probability of clouds

        # delete temporal folder if exists

        ...
