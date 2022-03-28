from typing import Union

import numpy as np

try:
    import gdal
    import gdalconst
    from gdal import Dataset
except ImportError:
    from osgeo import gdalconst
    from osgeo import gdal
    from osgeo.gdal import Dataset


def create_outfile_dataset(
    file_path: str,
    x_size: int,
    y_size: int,
    wkt_projection: Union[tuple, str],
    geo_transform: tuple,
    number_bands: int,
    driver: str = "GTiff",
    data_type=gdalconst.GDT_Int16,
    outfile_options: list = ["COMPRESS=DEFLATE"],
) -> Dataset:
    """Creates outfile dataset

    Uses GDAL to create an outfile Dataset to `file_path` using given metadata
    parameters

    Parameters
    ----------
    file_path : str
        Full file path to target raster file
    x_size : int
        Desired outfile dataset X size
    y_size : int
        Desired outfile dataset Y size
    wkt_projection : str
        WKT formated projection for outfile dataset
    geo_transform : tuple
        Geographic transformation for outfile dataset
    number_bands : int
        Number of bands for outfile dataset
    driver : str
        Outfile driver type. Default `GTiff`
    data_type : gdalconst.*
        Outfile data type. Default gdalconst.GDT_Int16
    outfile_options : list
        List of GDAL outfile options. Default ['COMPRESS=DEFLATE']

    Returns
    -------
    osgeo.gdal.Dataset
        GDAl dataset with given metdata parameters

    """

    # Create outfile driver
    gdal_driver = gdal.GetDriverByName(driver)

    # Create outfile dataset
    ds = gdal_driver.Create(
        file_path, x_size, y_size, number_bands, data_type, outfile_options
    )

    # Confirm successful `ds` creation
    if ds is None:
        raise TypeError(f"`ds` is None. GDAL was unable to create {file_path}")

    # Set outfile projection in WKT format
    ds.SetProjection(wkt_projection)

    # Set outfile geo transform
    ds.SetGeoTransform(geo_transform)

    return ds


def write_array_to_ds(
    ds: Dataset, array: np.ndarray, band: int = 1, no_data_value: int = -9999
) -> Dataset:
    """Writes NumPy array to GDAL Dataset band

    Uses GDAL to write `array` to `ds` `band` using given metadata parameters

    Parameters
    ----------
    ds : osgeo.gdal.Dataset
        GDAL dataset
    array : np.ndarray
        Full file path to target raster file
    band : int
        Target DS band to write `array`
    no_data_value : int
        No data value for `band`. Default -9999

    Returns
    -------
    osgeo.gdal.Dataset
        GDAl dataset with `array` written to `band`

    """

    # Confirm `ds` is valid
    if ds is None:
        raise ValueError(f"`ds` is None")

    number_bands: int = ds.RasterCount

    # Verify `band` is within the number of bands in `file_path` and
    # greater than zero
    if band > number_bands or band <= 0:
        raise ValueError(f"target band {band} is outside `ds` band scope")

    # Write `array` to outfile dataset
    ds.GetRasterBand(band).WriteArray(array)

    # Set outfile `no_data_value`
    ds.GetRasterBand(band).SetNoDataValue(no_data_value)

    return ds
