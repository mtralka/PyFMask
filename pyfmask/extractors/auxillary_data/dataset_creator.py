import logging.config
from pathlib import Path
from types import FunctionType
from typing import Any
from typing import Dict
from typing import Final
from typing import List
from typing import Union
from importlib import resources


import gdal
import osr
from pyfmask.extractors.auxillary_data.name_extractor import get_gswo_names
from pyfmask.extractors.auxillary_data.name_extractor import get_topo30_names
from pyfmask.extractors.auxillary_data.types import AuxTypes
from pyfmask.extractors.auxillary_data.types import BoundingBox
from pyfmask.extractors.auxillary_data.types import Coordinate


RESAMPLING_METHOD: Final[str] = "bilinear"

logger = logging.getLogger(__name__)


def create_local_aux_dataset(
    aux_path: Path,
    aux_type: AuxTypes,
    projection_reference: tuple,
    x_size: int,
    y_size,
    geo_transform: tuple,
    out_resolution: int,
    scene_id: str,
    no_data: Union[int, float],
    temp_dir: Path,
):

    original_upper_left: Coordinate = Coordinate(geo_transform[0], geo_transform[3])
    original_lower_right: Coordinate = Coordinate(
        geo_transform[0] + out_resolution * x_size,
        geo_transform[3] - out_resolution * y_size,
    )

    # transform to lat / lon
    proj_ref_lat_lon = osr.SpatialReference()
    proj_ref_lat_lon.ImportFromWkt(projection_reference)

    WGS_84 = osr.SpatialReference()
    WGS_84.ImportFromEPSG(4326)

    if int(gdal.VersionInfo()) >= 3000000:
        WGS_84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(proj_ref_lat_lon, WGS_84)
    (ul_lon, ul_lat, _) = transform.TransformPoint(
        original_upper_left.x, original_upper_left.y
    )
    (lr_lon, lr_lat, _) = transform.TransformPoint(
        original_lower_right.x, original_lower_right.y
    )

    bbox: BoundingBox = BoundingBox(
        NORTH=ul_lat, EAST=lr_lon, SOUTH=lr_lat, WEST=ul_lon
    )

    supported_platforms: dict = {
        AuxTypes.DEM: get_topo30_names,
        AuxTypes.GSWO: get_gswo_names,
    }

    get_names: FunctionType = supported_platforms[aux_type]

    aux_file_names: List[str] = get_names(bbox)

    logger.debug("Found aux files %s", aux_file_names)

    aux_dataset_list: List[Path] = []
    for file in aux_file_names:

        file_name_zip: str = file + ".zip"
        file_name_tif: str = file + ".tif"

        full_file_name_zip: Path = aux_path / file_name_zip

        if not full_file_name_zip.is_file():
            continue

        full_file_name_tif: Path = full_file_name_zip / file_name_tif

        aux_ds = gdal.Open("/vsizip/" + str(full_file_name_tif))

        if not aux_ds:
            continue

        aux_dataset_list.append(aux_ds)

    if len(aux_dataset_list) <= 0:
        logger.warn("No Aux DS files found")
        return None

    outfile_path: Path = temp_dir / f"_{scene_id}{aux_type.name}.tif"

    ds = gdal.Warp(
        str(outfile_path),
        aux_dataset_list,
        dstSRS=proj_ref_lat_lon,
        xRes=out_resolution,
        yRes=out_resolution,
        resampleAlg=RESAMPLING_METHOD,
        outputBounds=(
            original_upper_left.x,
            original_lower_right.y,
            original_lower_right.x,
            original_upper_left.y,
        ),
        srcNodata=no_data,
        dstNodata=no_data,
        format="GTiff",
    )

    if not ds:
        return None

    ds.GetRasterBand(1).SetNoDataValue(no_data)

    return ds


def create_mapzen_dataset(
    projection_reference: tuple,
    x_size: int,
    y_size,
    geo_transform: tuple,
    out_resolution: int,
    scene_id: str,
    no_data: Union[int, float],
    temp_dir: Path,
):
    """Create Mapzen GDAL DS using MAPZEN WMS file"""

    with resources.path("pyfmask.extractors.auxillary_data", "mapzen_wms.xml") as p:
        mapzen_wms: str = str(p)

    if not Path(mapzen_wms).is_file():
        logger.debug("No MAPZEN WMS file found")
        return None

    ##
    # Read DEM WMS
    ##
    dem_ds = gdal.Open(mapzen_wms)

    if not dem_ds:
        logger.warning("MAPZEN failed")
        return None

    original_upper_left: Coordinate = Coordinate(geo_transform[0], geo_transform[3])
    original_lower_right: Coordinate = Coordinate(
        geo_transform[0] + out_resolution * x_size,
        geo_transform[3] - out_resolution * y_size,
    )

    WGS_84 = osr.SpatialReference()
    WGS_84.ImportFromEPSG(4326)

    # transform to lat / lon
    proj_ref_lat_lon = osr.SpatialReference()
    proj_ref_lat_lon.ImportFromWkt(projection_reference)

    outfile_path: Path = temp_dir / f"_{scene_id}DEM.tif"

    ds = gdal.Warp(
        str(outfile_path),
        dem_ds,
        dstSRS=proj_ref_lat_lon,
        xRes=out_resolution,
        yRes=out_resolution,
        resampleAlg=RESAMPLING_METHOD,
        outputBounds=(
            original_upper_left.x,
            original_lower_right.y,
            original_lower_right.x,
            original_upper_left.y,
        ),
        srcNodata=no_data,
        dstNodata=no_data,
        format="GTiff",
    )

    if not ds:
        logger.warning("MAPZEN failed")
        return None

    ds.GetRasterBand(1).SetNoDataValue(no_data)
    dem_ds = None

    return ds
