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
        "--save-cloud-prob",
        help="Boolean whether to output cloud probability map",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--dem-path", help="Path to DEM where folder GTOPO30ZIP located"
    )
    parser.add_argument(
        "--gswo-path", help="Path to GWSO where folder GSWO150ZIP located"
    )
    args = parser.parse_args()

    fmask_control = FMask(
        infile=args.get("infile"),
        out_dir=args.get("out_dir"),
        out_name=args.get("out_name"),
        dem_path=args.get("dem-path"),
        gswo_path=args.get("gswo-path"),
        dilated_shadow_px=args.get("shadow"),
        dilated_cloud_px=args.get("cloud"),
        dilated_snow_px=args.get("snow"),
        save_cloud_prob=args.get("save-cloud-prob"),
        auto_save=True,
        auto_run=True,
        delete_temp_dir=True,
    )

    return 0


if __name__ == "__main__":
    app()
