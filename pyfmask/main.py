from importlib import resources
import json
import logging.config
from pathlib import Path
from shutil import rmtree
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from typing import cast
import time

import gdalconst
import numpy as np

from pyfmask.detectors import detect_absolute_snow
from pyfmask.detectors import detect_snow
from pyfmask.detectors import detect_water
from pyfmask.detectors.cloud import detect_false_positive_cloud_pixels
from pyfmask.detectors.cloud import detect_potential_cloud_pixels
from pyfmask.detectors.cloud import detect_potential_clouds
from pyfmask.detectors.cloud_shadow import detect_potential_cloud_shadow_pixels
from pyfmask.detectors.cloud_shadow import match_cloud_shadows
from pyfmask.extractors.auxillary_data import AuxTypes
from pyfmask.extractors.auxillary_data import extract_aux_data
from pyfmask.platforms.landsat8 import Landsat8
from pyfmask.platforms.sentinel2 import Sentinel2
from pyfmask.raster_utilities.composites import create_cdi
from pyfmask.raster_utilities.composites import create_ndbi
from pyfmask.raster_utilities.composites import create_ndsi
from pyfmask.raster_utilities.composites import create_ndvi
from pyfmask.raster_utilities.io import create_outfile_dataset
from pyfmask.raster_utilities.io import write_array_to_ds
from pyfmask.raster_utilities.morphology import dilate_array
from pyfmask.raster_utilities.morphology import enhance_line
from pyfmask.raster_utilities.morphology import erode_commissons
from pyfmask.classes import DEMData
from pyfmask.classes import GSWOData
from pyfmask.classes import PotentialCloudPixels
from pyfmask.classes import PotentialClouds
from pyfmask.classes import SensorData
from pyfmask.utils import valdiate_path

with resources.path("pyfmask", "loggingConfig.json") as p:
    logging_config_path: str = str(p)

with open(logging_config_path, "rt") as file:
    logging.config.dictConfig(json.load(file))

logger = logging.getLogger(__name__)


class FMask:
    def __init__(
        self,
        infile: Union[Path, str],
        out_dir: Union[Path, str],
        out_name: str,
        gswo_path: Union[Path, str],
        dem_path: Optional[Union[Path, str]] = None,
        dilated_cloud_px: int = 3,
        dilated_shadow_px: int = 3,
        dilated_snow_px: int = 0,
        dem_nodata: Union[float, int] = -9999,
        gswo_nodata: Union[float, int] = 255,
        use_mapzen: bool = True,
        auto_run: bool = False,
        auto_save: bool = True,
        delete_temp_dir: bool = True,
        save_cloud_prob: bool = True,
    ):
        self.infile: Path = valdiate_path(infile, check_exists=True, check_is_file=True)

        self.out_dir: Path = valdiate_path(out_dir, check_is_dir=True)
        self.out_name: str = out_name

        self.gswo_path: Path = valdiate_path(
            gswo_path, check_exists=True, check_is_dir=True
        )

        self.dem_path: Optional[Path] = None
        if dem_path:
            self.dem_path = valdiate_path(
                dem_path, check_exists=True, check_is_dir=True
            )

        if not use_mapzen and not dem_path:
            logger.error(
                "Must select `use_mapzen` or provide a local `dem_path`",
                stack_info=True,
            )
            raise ValueError("Must select `use_mapzen` or provide a local `dem_path`")

        self.use_mapzen: bool = use_mapzen

        self.auto_save: bool = auto_save
        self.delete_temp_dir: bool = delete_temp_dir

        self.dilated_cloud_px: int = dilated_cloud_px
        self.dilated_shadow_px: int = dilated_shadow_px
        self.dilated_snow_px: int = dilated_snow_px

        self.save_cloud_prob: bool = save_cloud_prob

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

        if auto_run:
            self.run()

    @property
    def outfile_path(self) -> str:
        return str(self.out_dir / self.out_name)

    def run(self) -> None:

        logger.info("Starting FMask")

        start_time: float = time.time()

        ##
        # Extract platform data
        ##
        self._extract_platform_data()

        ##
        # Create temporary directory for intermediate files
        ##
        self.temp_dir = self._create_temp_directory()

        ##
        # Extract DEM data
        ##
        self._extract_dem_data()

        ##
        # Create requisite spectral composites
        ##
        self._create_spectral_composites()

        ##
        # Detect snow area
        ##
        logger.info("Starting snow detection")
        self.snow = detect_snow(
            self.ndsi,
            band_data=self.platform_data.band_data,
        )

        ##
        # Detect water area
        ##
        logger.info("Starting water detection")
        self.water, self.all_water = detect_water(
            self.platform_data.band_data["NIR"],
            self.ndvi,
            self.platform_data.nodata_mask,
            self.snow,
            cast(GSWOData, self.gswo_data),
        )

        ##
        # Calculate CDI if platform is S2
        ##
        if self.platform_data.sensor == "S2_MSI":
            self.cdi = create_cdi(
                self.platform_data.band_data["NIR"],
                self.platform_data.band_data["NIR2"],
                self.platform_data.band_data["RED3"],
            )

        ##
        # Detect absolute snow (pure snow / ice)
        ##
        logger.info("Starting absolute snow detection")
        self.absolute_snow = detect_absolute_snow(
            self.platform_data.sensor,
            self.snow,
            self.platform_data.band_data,
            self.platform_data.vis_saturation,
            self.ndsi,
        )

        ##
        # Detect cloud area
        ##
        self._run_cloud_detection()

        ##
        # Detect cloud shadow area
        ##
        self._run_cloud_shadow_detection()

        ##
        # Dilate snow, shadows, and clouds
        ##
        if self.dilated_snow_px > 0:
            self.snow = dilate_array(self.snow, self.dilated_snow_px)
        if self.dilated_shadow_px > 0:
            self.cloud_shadow = dilate_array(self.cloud_shadow, self.dilated_shadow_px)
        if self.dilated_cloud_px > 0:
            self.cloud = dilate_array(self.cloud, self.dilated_cloud_px)

        ##
        # Compute final results
        ##
        self._compute_final_results()

        logger.info("Completed FMask run in %s seconds", time.time() - start_time)

        if self.auto_save:
            self.save_results()

        if self.delete_temp_dir:
            logger.debug("Removing temporary directory at %s", self.temp_dir)
            rmtree(self.temp_dir, ignore_errors=True)

        return None

    def _extract_platform_data(self) -> None:
        """Extract platform data"""

        supported_platforms: Dict[str, Any] = {
            "Landsat8": Landsat8,
            "Sentinel2": Sentinel2,
        }

        for name, platform_object in supported_platforms.items():
            if platform_object.is_platform(self.infile):

                logging.info("Identified as %s", name)

                self.platform_data = platform_object.get_data(self.infile)
                break
        else:
            logger.error("Platform not found or supported", stack_info=True)
            raise ValueError("Platform not found or supported")

        return None

    def _extract_dem_data(self) -> None:
        """Extract data from DEMs and GSWOs"""

        logger.info("Extracting auxillary DEM and GSWO Data")

        aux_data_kwargs: Dict[str, Any] = {
            "projection_reference": self.platform_data.projection_reference,
            "x_size": self.platform_data.x_size,
            "y_size": self.platform_data.y_size,
            "geo_transform": self.platform_data.geo_transform,
            "out_resolution": self.platform_data.out_resolution,
            "scene_id": self.platform_data.scene_id,
            "temp_dir": self.temp_dir,
        }

        initial_dem_type: AuxTypes = (
            AuxTypes.MAPZEN if self.use_mapzen else AuxTypes.DEM
        )
        dem_data = extract_aux_data(
            aux_path=self.dem_path,
            aux_type=initial_dem_type,
            no_data=self.dem_nodata,
            **aux_data_kwargs,
        )

        ##
        # Retry DEM collection with local if MAPZEN failed
        # and local `self.dem_data` path was given
        ##
        if (
            not dem_data
            and initial_dem_type is AuxTypes.MAPZEN
            and self.dem_data is not None
        ):
            logger.info("Failed Mapzen, using local DEM")
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

        return None

    def _create_spectral_composites(self) -> None:
        """Create `ndvi`, `ndsi`, and `ndbi`"""

        logger.info("Creating spectral composites")

        self.ndvi = create_ndvi(
            self.platform_data.band_data["RED"], self.platform_data.band_data["NIR"]
        )
        self.ndsi = create_ndsi(
            self.platform_data.band_data["GREEN"], self.platform_data.band_data["SWIR1"]
        )
        self.ndbi = create_ndbi(
            self.platform_data.band_data["SWIR1"], self.platform_data.band_data["NIR"]
        )

        return None

    def _create_temp_directory(self) -> Path:
        """Create temporary directory"""

        temp_name: str = str(self.platform_data.scene_id) + "_temp"
        outfile_path: Path = self.out_dir / temp_name

        if outfile_path.exists():
            logger.debug("Outfile path exists, rewriting")

        outfile_path.mkdir(exist_ok=True)

        return outfile_path

    def _run_cloud_detection(self) -> None:
        """Run cloud detection pipeline"""

        logger.info("Starting cloud detection")

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

        ##
        # Update cirrus clouds
        ##
        if self.platform_data.band_data.get("CIRRUS") is not None:
            self.platform_data.band_data["CIRRUS"] = cast(
                np.ndarray, self.potential_cloud_pixels.normalized_cirrus
            )

        ##
        # Detect potential clouds
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
        potential_false_positives: np.ndarray = detect_false_positive_cloud_pixels(
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

        logger.debug("Finished cloud detection")

        return None

    def _run_cloud_shadow_detection(self, pixel_limit: int = 40000) -> None:
        """Run cloud shadow detection pipeline"""

        logger.info("Starting cloud shadow detection")

        if self.potential_clouds.sum_clear_pixels < pixel_limit:

            logger.debug("No clear pixels in image")
            self.cloud_shadow = np.where(self.cloud == 0, 1, 0)
            return

        ##
        # Find potential cloud shadow pixels
        ##
        potential_cloud_shadow_pixels: np.ndarray = (
            detect_potential_cloud_shadow_pixels(
                self.platform_data, self.dem_data, self.potential_clouds.clear_land
            )
        )

        ##
        # Match potential cloud shadow pixels to clouds
        ##
        self.cloud_shadow = match_cloud_shadows(
            self.cloud,
            self.potential_clouds.sum_clear_pixels,
            self.all_water,
            potential_cloud_shadow_pixels,
            self.platform_data,
            self.dem_data,
            self.potential_clouds.temp_test_low,
            self.potential_clouds.temp_test_high,
        )

        logger.debug("Finished cloud shadow detection")

        return None

    def _compute_final_results(self) -> None:
        """Compute final fmask array results"""

        final_array: np.ndarray = np.zeros(
            self.potential_clouds.cloud.shape, dtype=np.uint8
        )

        ##
        # Water
        ##
        final_array = np.where(self.water > 0, 1, final_array)

        ##
        # Snow
        ##
        final_array = np.where(self.snow > 0, 3, final_array)

        ##
        # Cloud Shadow
        ##
        final_array = np.where(self.cloud_shadow > 0, 2, final_array)

        ##
        # Cloud
        ##
        final_array = np.where(self.cloud > 0, 4, final_array)

        ##
        # No Data
        ##
        final_array = np.where(self.platform_data.nodata_mask, 255, final_array)

        self.results = final_array

        return None

    def save_results(self) -> None:
        """Save `self.final_array` to `self.outfile_path`"""

        logger.info("Saving results")

        ##
        # Save fmask `self.results`
        ##
        self.save_array_to_file(
            self.results,
            self.outfile_path,
        )

        ##
        # [Optional] Save cloud probability
        ##
        logger.debug("Saving cloud probability %s", self.save_cloud_prob)
        if self.save_cloud_prob:
            cloud_probability: np.ndarray = np.where(
                self.water == 1,
                self.potential_clouds.over_water_probability,
                self.potential_clouds.over_land_probability,
            )
            cloud_probability = np.where(cloud_probability < 0, 0, cloud_probability)
            cloud_probability = np.where(
                cloud_probability > 100, 100, cloud_probability
            )
            cloud_probability = np.where(
                self.platform_data.nodata_mask, 255, cloud_probability
            )

            outfile_path = self.outfile_path
            if outfile_path[-4:] == ".tif":
                outfile_path = self.outfile_path[0:-4]

            outfile_path += "_cloud-probability.tif"

            self.save_array_to_file(
                cloud_probability,
                outfile_path,
            )

        return None

    def save_array_to_file(
        self,
        array: np.ndarray,
        file_path: str,
    ) -> None:
        """Save `array` to `file_path` using `self` raster paramters"""

        x_size: int = self.potential_clouds.cloud.shape[1]
        y_size: int = self.potential_clouds.cloud.shape[0]

        outfile_ds = create_outfile_dataset(
            file_path,
            x_size,
            y_size,
            self.platform_data.projection_reference,
            self.platform_data.geo_transform,
            1,
            data_type=gdalconst.GDT_Byte,
        )
        outfile_ds = write_array_to_ds(outfile_ds, array)
        outfile_ds = None

        logger.debug("Saved array to %s", file_path)

        return None
