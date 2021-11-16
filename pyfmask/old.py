import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        "Fmask python version 4.3 for Landsat 8 and Sentinel-2"
    )
    parser.add_argument(
        "input", help="input path to file *_MTL.txt (L8) or MTD_TL.xml (S2)"
    )
    parser.add_argument("output", help="output path where cloud mask will be saved")
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
        "--p",
        help="Cloud probability threshold. Default values: L8=17.5, S2=20",
        type=float,
    )
    parser.add_argument(
        "--prob_output",
        help="Boolean value (0 or 1=output) whether to output cloud probability map (0-100)",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--path_dem", help="Path to DEM where folder GTOPO30ZIP located"
    )
    parser.add_argument(
        "--path_gswo", help="Path to GWSO where folder GSWO150ZIP located"
    )
    args = parser.parse_args()

    # TBD: add checking errors on the existance of input and output

    t0 = time.time()
    print(args)
    print(args.input)
    # fmask(args)
    print(f"Processing time: {time.time()-t0} sec")

    return 0


if __name__ == "__main__":
    main()
