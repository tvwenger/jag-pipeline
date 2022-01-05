"""
flagbackup.py
Backup the flag table to an external file.

Copyright(C) 2021 by
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
Trey Wenger - December 2021
"""

import os
import argparse
import h5py
import numpy as np

from . import __version__


def flagbackup(datafile, backupfile):
    """
    Save the flag table to an external hd5 file.

    Inputs:
        datafile :: string
            SDHDF filename
        backupfile :: string
            Flag backup filename

    Returns: Nothing
    """
    if os.path.exists(backupfile):
        raise FileExistsError(f"Will not overwrite {backupfile}")

    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r", rdcc_nbytes=cache_size) as sdhdf:
        with h5py.File(backupfile, "w") as backuphdf:
            # Loop over scans
            scans = [
                key
                for key in sdhdf["data"]["beam_0"]["band_SB0"].keys()
                if "scan" in key
            ]
            scans = sorted(scans, key=lambda scan: int(scan[5:]))
            for scan in scans:
                flag = sdhdf["data"]["beam_0"]["band_SB0"][scan]["flag"]
                data = np.empty((0, flag.shape[1]), dtype=bool)
                flagbackup = backuphdf.create_dataset(
                    scan, data=data, maxshape=(None, flag.shape[1])
                )
                for i in range(flag.shape[0]):
                    flagbackup.resize(flagbackup.shape[0] + 1, axis=0)
                    flagbackup[-1] = flag[i]


def main():
    parser = argparse.ArgumentParser(
        description="Save the flag table to an external hd5 file.",
        prog="flagbackup.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    parser.add_argument(
        "backupfile", type=str, help="Flag backup HDF file",
    )
    args = parser.parse_args()
    flagbackup(args.datafile, args.backupfile)


if __name__ == "__main__":
    main()
