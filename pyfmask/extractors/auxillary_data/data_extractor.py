from pathlib import Path
from types import FunctionType
from typing import Any
from typing import Dict
from typing import Final
from typing import List

from typing import Optional

from typing import Union

import gdal
import osr
from pyfmask.auxillary_data_extractor.dataset_creator import create_aux_dataset
from pyfmask.auxillary_data_extractor.types import AuxTypes
from pyfmask.utils.classes import DEMData
from pyfmask.utils.classes import GSWOData


RESAMPLING_METHOD: Final[str] = "bilinear"


def extract_aux_data(
    aux_path: Path,
    aux_type: AuxTypes,
    projection_reference: tuple,
    x_size: int,
    y_size,
    geo_transform: tuple,
    out_resolution: int,
    scene_id: str,
    no_data: Union[int, float],
) -> Optional[Union[DEMData, GSWOData]]:

    if not isinstance(aux_type, AuxTypes):
        raise ValueError(
            f"`aux_type` must be one of {','.join([a.name for a in AuxTypes])}"
        )

    ds = create_aux_dataset(
        aux_path,
        aux_type,
        projection_reference,
        x_size,
        y_size,
        geo_transform,
        out_resolution,
        scene_id,
        no_data,
    )

    if ds is None:
        return None

    supported_platforms: Dict[Any, FunctionType] = {
        AuxTypes.DEM: extract_dem_data,
        AuxTypes.GSWO: extract_gswo_data,
    }

    extractor_function: FunctionType = supported_platforms[aux_type]

    data: Union[DEMData, GSWOData] = extractor_function(ds)

    ds = None

    return data


def extract_dem_data() -> DEMData:

    # calc dem, slop, aspect
    # return DEMData
    ...


def extract_gswo_data() -> GSWOData:
    # calc gdwo
    # return GSWOData

    ...
