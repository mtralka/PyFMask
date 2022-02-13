from pyfmask.extractors.auxillary_data.types import BoundingBox


def get_topo30_names(bbox: BoundingBox) -> list:
    fname_list = []
    north = bbox.NORTH
    south = bbox.SOUTH
    west = bbox.WEST
    east = bbox.EAST

    # Processing North-South Lat data 90..-90
    idx_north = int((90 - north) / 50.0)
    idx_south = int((90 - south) / 50.0)

    fname_list_tmp = []
    for k in range(idx_north, idx_south + 1):
        idx_lat = 90 - 50 * k

        if idx_lat > 0:
            idx_str = f"n{idx_lat}"
        else:
            idx_str = f"s{abs(idx_lat)}"
        fname_list_tmp.append(idx_str)

    # Processing West-East Lon -180..180
    for f_north in fname_list_tmp:
        if "s60" in f_north:
            step = 60.0
        else:
            step = 40.0
        idx_west = int((west + 180) / step)
        idx_east = int((east + 180) / step)

        if west <= east:
            idx_list = [*range(idx_west, idx_east + 1)]
        else:
            # Processing a situation when west > east
            # this happens when corssing 180E and 180W line
            # there can be both Landsat and Sentinel scenes
            # from west to 180E
            max_idx = int((180 + 180) / step)
            min_idx = 0
            idx_list = [*range(idx_west, max_idx), *range(0, idx_east + 1)]

        for k in idx_list:
            idx_lon = int(-180 + step * k)

            if idx_lon <= 0:
                idx_str = "w%03d" % (abs(idx_lon))
            else:
                idx_str = "e%03d" % (idx_lon)
                f_str = f"gt30{idx_str}{f_north}"
                fname_list.append(f_str)

    return fname_list


def get_gswo_names(bbox: BoundingBox) -> list:
    fname_list = []
    north = bbox.NORTH
    south = bbox.SOUTH
    west = bbox.WEST
    east = bbox.EAST

    # Processing North-South Lat data 80..-80 (or UL 80..-70)
    step = 10.0
    idx_north = int((90 - north) / step)
    idx_south = int((90 - south) / step)

    fname_list_tmp = []
    for k in range(idx_north, idx_south + 1):
        idx_lat = int(90 - step * k)
        # only 80..-80 available
        if (idx_lat == 90) | (idx_lat == -80):
            continue

        if idx_lat >= 0:
            idx_str = f"{idx_lat}N"
        else:
            idx_str = f"{abs(idx_lat)}S"
        fname_tmp = f"{idx_str}"
        fname_list_tmp.append(fname_tmp)

    # Processing West-East Lon -180..180 (or UL-180..170)
    step = 10
    for f_north in fname_list_tmp:
        idx_west = int((west + 180) / step)
        idx_east = int((east + 180) / step)

        if west <= east:
            idx_list = [*range(idx_west, idx_east + 1)]
        else:
            # Processing a situation when west > east
            # this happens when corssing 180E and 180W line
            # there can be both Landsat and Sentinel scenes
            # from west to 180E
            max_idx = int((180 + 180) / step)
            min_idx = 0
            idx_list = [*range(idx_west, max_idx), *range(0, idx_east + 1)]

        for k in idx_list:
            idx_lon = int(-180 + step * k)

            if idx_lon < 0:
                idx_str = f"{abs(idx_lon)}W"
            else:
                idx_str = f"{idx_lon}E"
                fname = f"occurrence_{idx_str}_{f_north}"
                fname_list.append(fname)

    return fname_list
