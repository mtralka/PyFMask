from argparse import ArgumentParser

from pyfmask.main import FMask


def app():
    parser = ArgumentParser("PyFMask4.3 Landsat 8 and Sentinel-2")
    parser.add_argument(
        "infile", help="infile path to *_MTL.txt (L8) or MTD_TL.xml (S2) files"
    )
    parser.add_argument("out_dir", help="output directory for fmask results")
    parser.add_argument("out_name", help="output file name for fmask file")
    parser.add_argument(
        "--cloud",
        help="Dilated number of pixels for cloud, default value of 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--shadow",
        help="Dilated number of pixels for cloud shadow, default value of 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--snow",
        help="Dilated number of pixels for snow, default value of 0",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_cloud_prob",
        help="Boolean whether to output cloud probability map",
        type=bool,
        default=False,
    ),
    parser.add_argument(
        "--use_mapzen",
        help="Boolean to use Mapzen",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--dem_path", help="Path to DEM where folder GTOPO30ZIP located"
    )
    parser.add_argument(
        "--gswo_path", help="Path to GWSO where folder GSWO150ZIP located"
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
