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
        "--cloud",
        help="Number of cloud pixels to dilate, default 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--shadow",
        help="Number of cloud shadow pixels to dilate, default 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--snow",
        help="Number of snow pixels to dilate, default to 0",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_cloud_prob",
        help="Boolean whether to output cloud probability map, default of True",
        type=bool,
        default=True,
    ),
    parser.add_argument(
        "--use_mapzen",
        help="Bool to use Mapzen WMS DEM mapping, default True",
        type=bool,
        default=True,
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
    args = parser.parse_args()

    fmask_control = FMask(
        infile=args.infile,
        out_dir=args.out_dir,
        out_name=args.out_name,
        dem_path=args.dem_path,
        gswo_path=args.gswo_path,
        dilated_shadow_px=args.shadow,
        dilated_cloud_px=args.cloud,
        dilated_snow_px=args.snow,
        save_cloud_prob=args.save_cloud_prob,
        auto_save=True,
        auto_run=True,
        delete_temp_dir=True,
        use_mapzen=args.use_mapzen,
    )

    return 0


if __name__ == "__main__":
    app()
