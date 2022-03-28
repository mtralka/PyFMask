from argparse import ArgumentParser

from pyfmask.main import FMask


def app():
    parser = ArgumentParser("PyFMask Landsat 8 and Sentinel-2")
    parser.add_argument(
        "infile",
        help=" Path to Sentinel-2 or Landast-8 file EX. {*._MTL.txt, MTD_*.xml}",
        type=str,
    )
    parser.add_argument(
        "out_dir",
        help="Directory for program outputs EX. temporary folder, fmask results, probability masks",
    )
    parser.add_argument(
        "--out_name",
        help="Override default naming method `[self.platform_data.scene_id]_fmask`, default None",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dem_path",
        help="Optional path to local DEM directory GTOPO30ZIP, default None",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gswo_path",
        help="Optional path to local GSWO directory GSWO150ZIP, default None",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dilated_cloud_px",
        help="Number of cloud pixels to dilate, default 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--dilated_shadow_px",
        help="Number of cloud shadow pixels to dilate, default 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--dilated_snow_px",
        help="Number of snow pixels to dilate, default to 0",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dem_nodata",
        help="Override for standard DEM nodata values, default -9999",
        type=int,
        default=-9999,
    )
    parser.add_argument(
        "--gswo_nodata",
        help="Override for standard GSWO nodata values, default 255",
        type=int,
        default=255,
    )
    parser.add_argument(
        "--water_value",
        help="Value for water in fmask result array, default 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--snow_value",
        help="Value for snow in fmask result array, default 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--cloud_shadow_value",
        help="Value for cloud shadow in fmask result array, default 2",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--cloud_value",
        help="Value for cloud in fmask result array, default 4",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--use_mapzen",
        help="Bool to use Mapzen WMS DEM mapping, default True",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--delete_temp_dir",
        help="Bool to automatically delete temporary directory - `temp_dir`, default True",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--save_cloud_prob",
        help="Boolean whether to output cloud probability map, default of True",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--log_in_temp_dir",
        help="Bool to update log file locations to `temp_dir` after `temp_dir` creation, default True",
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    _ = FMask(
        infile=args.infile,
        out_dir=args.out_dir,
        out_name=args.out_name,
        dem_path=args.dem_path,
        gswo_path=args.gswo_path,
        dilated_shadow_px=args.dilated_shadow_px,
        dilated_cloud_px=args.dilated_cloud_px,
        dilated_snow_px=args.dilated_snow_px,
        dem_nodata=args.dem_nodata,
        gswo_nodata=args.gswo_nodata,
        water_value=args.water_value,
        snow_value=args.snow_value,
        cloud_shadow_value=args.cloud_shadow_value,
        cloud_value=args.cloud_value,
        use_mapzen=args.use_mapzen,
        delete_temp_dir=args.delete_temp_dir,
        save_cloud_prob=args.save_cloud_prob,
        log_in_temp_dir=args.log_in_temp_dir,
        auto_save=True,
        auto_run=True,
    )

    return 0


if __name__ == "__main__":
    app()
