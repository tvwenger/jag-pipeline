"""
flagsummary.py
Print flag summary.

Copyright(C) 2021-2023 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changelog:
Trey Wenger - January 2022
"""

import argparse
import h5py
import numpy as np

from . import __version__


def flagsummary(datafile):
    """
    Print summary of flags.

    Inputs:
        datafile :: string
            SDHDF filename

    Returns: Nothing
    """
    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    num_data = 0
    num_flagged = 0
    with h5py.File(datafile, "r", rdcc_nbytes=cache_size) as sdhdf:
        # Loop over scans
        scans = [
            key for key in sdhdf["data"]["beam_0"]["band_SB0"].keys() if "scan" in key
        ]
        scans = sorted(scans, key=lambda scan: int(scan[5:]))
        for scan in scans:
            scan_data = 0
            scan_flagged = 0
            flag = sdhdf["data"]["beam_0"]["band_SB0"][scan]["flag"]
            source = sdhdf["data"]["beam_0"]["band_SB0"][scan].attrs["SOURCE"]
            for i in range(flag.shape[0]):
                scan_data += flag[i].size
                scan_flagged += np.sum(flag[i])
            print(
                f"{scan} [{source}]: {scan_flagged}/{scan_data} "
                + f"({100.0*scan_flagged/scan_data:0.2f}%) flagged"
            )
            num_data += scan_data
            num_flagged += scan_flagged
        print(
            f"Total: {num_flagged}/{num_data} "
            + f"({100.0*num_flagged/num_data:0.2f}%) flagged"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Print flag summary",
        prog="flagsummary.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    args = parser.parse_args()
    flagsummary(args.datafile)


if __name__ == "__main__":
    main()
