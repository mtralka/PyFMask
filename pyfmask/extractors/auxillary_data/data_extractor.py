import logging.config
from multiprocessing.sharedctypes import Value
from pathlib import Path
from types import FunctionType
from typing import Any
from typing import Final
from typing import Optional
from typing import Union
from typing import cast

import gdal
import numpy as np
import osr
from pyfmask.extractors.auxillary_data.dataset_creator import create_local_aux_dataset
from pyfmask.extractors.auxillary_data.dataset_creator import create_mapzen_dataset
from pyfmask.extractors.auxillary_data.types import AuxTypes
from pyfmask.classes import DEMData
from pyfmask.classes import GSWOData


logger = logging.getLogger(__name__)

RESAMPLING_METHOD: Final[str] = "bilinear"


def extract_aux_data(
    aux_path: Optional[Path],
    aux_type: AuxTypes,
    projection_reference: tuple,
    x_size: int,
    y_size,
    geo_transform: tuple,
    out_resolution: int,
    scene_id: str,
    no_data: Union[int, float],
    temp_dir: Path,
) -> Optional[Union[DEMData, GSWOData]]:

    if not isinstance(aux_type, AuxTypes):
        raise ValueError(
            f"`aux_type` must be one of {','.join([a.name for a in AuxTypes])}"
        )

    if aux_type == AuxTypes.MAPZEN:
        logger.info("Using MAPZEN DEM")
        ds = create_mapzen_dataset(
            projection_reference,
            x_size,
            y_size,
            geo_transform,
            out_resolution,
            scene_id,
            no_data,
            temp_dir,
        )
    elif aux_type is not AuxTypes.MAPZEN and aux_path is not None:
        ds = create_local_aux_dataset(
            cast(Path, aux_path),
            aux_type,
            projection_reference,
            x_size,
            y_size,
            geo_transform,
            out_resolution,
            scene_id,
            no_data,
            temp_dir,
        )
    else:
        return None

    if ds is None:
        return None

    data_extractors: dict = {
        AuxTypes.DEM: extract_dem_data,
        AuxTypes.MAPZEN: extract_dem_data,
        AuxTypes.GSWO: extract_gswo_data,
    }

    extractor_function: FunctionType = data_extractors[aux_type]

    data: Union[DEMData, GSWOData] = extractor_function(
        ds, scene_id=scene_id, temp_dir=temp_dir
    )
    ds = None

    return data


def extract_dem_data(ds, scene_id: str, temp_dir: Path) -> DEMData:

    dem_arr: np.ndarray = ds.GetRasterBand(1).ReadAsArray()

    slope_name: str = f"{scene_id}_slope.tif"
    slope_ds = gdal.DEMProcessing(
        str(temp_dir / slope_name), ds, processing="slope", slopeFormat="degree"
    )
    slope_arr: np.ndarray = slope_ds.GetRasterBand(1).ReadAsArray()
    slope_ds = None

    aspect_name: str = f"{scene_id}_aspect.tif"
    aspect_ds = gdal.DEMProcessing(
        str(temp_dir / aspect_name), ds, processing="aspect", zeroForFlat=True
    )
    aspect_arr: np.ndarray = aspect_ds.GetRasterBand(1).ReadAsArray()
    aspect_ds = None

    ds = None

    logger.debug("Retrieved DEM from %s", str(temp_dir / aspect_name))

    return DEMData(dem=dem_arr, slope=slope_arr, aspect=aspect_arr)


def extract_gswo_data(ds, **kwargs: Any) -> GSWOData:

    gswo_arr: np.ndarray = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    #  255 is 100% ocean
    filter_arr: np.ndarray = np.where(gswo_arr == 255, 100, gswo_arr)

    return GSWOData(gswo=filter_arr)
