from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from typing import cast

from pyfmask.extractors.auxillary_data import AuxTypes
from pyfmask.extractors.auxillary_data import extract_aux_data
from pyfmask.platforms.landsat8 import Landsat8
from pyfmask.utils.classes import DEMData
from pyfmask.utils.classes import GSWOData
from pyfmask.utils.classes import SensorData


class fmask:
    def __init__(
        self,
        infile: Union[Path, str],
        out_dir: Union[Path, str],
        dem_path: Union[Path, str],
        gswo_path: Union[Path, str],
        cloud_threshold: Optional[Union[float, int]] = None,
        dilated_cloud_px: int = 3,
        dilated_shadow_px: int = 3,
        dialted_snow_px: int = 0,
        output_cloud_prob: bool = True,
        dem_nodata: Union[float, int] = -9999,
        gswo_nodata: Union[float, int] = 255,
    ):
        self.infile: Path = self._valdiate_path(
            infile, check_exists=True, check_is_file=True
        )
        self.out_dir: Path = self._valdiate_path(out_dir, check_is_dir=True)
        self.dem_path: Path = self._valdiate_path(
            dem_path, check_exists=True, check_is_dir=True
        )
        self.gswo_path: Path = self._valdiate_path(
            gswo_path, check_exists=True, check_is_dir=True
        )

        if cloud_threshold:
            self.cloud_threshold: float = float(cloud_threshold)

        self.dilated_cloud_px: int = dilated_cloud_px
        self.dilated_shadow_px: int = dilated_shadow_px
        self.dialted_snow_px: int = dialted_snow_px

        self.output_cloud_prob: bool = output_cloud_prob

        self.dem_nodata: Union[float, int] = dem_nodata
        self.gswo_nodata: Union[float, int] = gswo_nodata

        self.platform_data: SensorData
        self.dem_data: Optional[DEMData]
        self.gwso_data: Optional[GSWOData]

    def run(self) -> None:

        self.platform_data = self.extract_platform_data()
        
        self.temp_dir: Path = self._create_temp_directory()
        print("CREATED TEMP DIR")
        aux_data_kwargs: Dict[str, Any] = {
            "projection_reference": self.platform_data.projection_reference,
            "x_size": self.platform_data.x_size,
            "y_size": self.platform_data.y_size,
            "geo_transform": self.platform_data.geo_transform,
            "out_resolution": self.platform_data.out_resolution,
            "scene_id": self.platform_data.scene_id,
            "temp_dir": self.temp_dir
        }

        dem_data = extract_aux_data(
            aux_path=self.dem_path,
            aux_type=AuxTypes.DEM,
            no_data=self.dem_nodata,
            **aux_data_kwargs,
        )
        print("DEM DATA")
        self.dem_data = cast(DEMData, dem_data) if dem_data else None

        gwso_data = extract_aux_data(
            aux_path=self.gswo_path,
            aux_type=AuxTypes.GSWO,
            no_data=self.gswo_nodata,
            **aux_data_kwargs,
        )
        print("GSWO DATA")
        self.gwso_data = cast(GSWOData, gwso_data) if gwso_data else None

        # calc NDVI

        # calc NSDI

        # calc NDBI

        # detect snow

        # detect water

        # calc cdi if s2

        # detect pot clouds pixels

        # if cirrus band
        # update normalized

        # deect abs snow ice

        # add ndbi index

        # detect pot clouds

        # if bt band and bt dem

        # detect fallse positivers

        # erode commison

        # detect cloud shadow

        # dilalte, snow, shadow, cloud

        # save
        ...

    @staticmethod
    def _valdiate_path(
        path: Union[str, Path],
        check_exists: bool = False,
        check_is_file: bool = False,
        check_is_dir=False,
    ) -> Path:

        valid_path: Path = Path(path) if isinstance(path, str) else path

        if check_exists:
            if not valid_path.exists():
                raise FileExistsError(f"{path} must exist")

        if check_is_file:
            if not valid_path.is_file():
                raise FileNotFoundError(f"{path} must be a file")

        if check_is_dir:
            if not valid_path.is_dir():
                raise ValueError(f"{path} must be a directory")

        return valid_path

    def extract_platform_data(self) -> SensorData:

        supported_platforms: Dict[str, Any] = {"Landsat8": Landsat8}

        for name, platform_object in supported_platforms.items():
            if platform_object.is_platform(self.infile):
                print(f"Identified as {name}")
                return platform_object.get_data(self.infile)
        else:
            raise ValueError("Platform not found or supported")

    def _create_temp_directory(self) -> Path:

        temp_name: str = str(self.platform_data.scene_id) + "_temp"
        outfile_path: Path = self.out_dir / temp_name

        if outfile_path.exists():
            print("WARN, outfile path exists, rewriting")

        outfile_path.mkdir(exist_ok=True)

        return outfile_path
