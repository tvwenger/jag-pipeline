"""
flagrestore.py
Restore flags from backup file.

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

import argparse
import h5py

from . import __version__


def flagrestore(datafile, backupfile):
    """
    Restore flags from backup file.

    Inputs:
        datafile :: string
            SDHDF filename
        backupfile :: string
            Flag backup filename

    Returns: Nothing
    """
    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        flag = sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["flag"]

        with h5py.File(backupfile, "r") as backuphdf:
            flagbackup = backuphdf["flagbackup"]

            if flag.shape != flagbackup.shape:
                raise ValueError("Conflicting flag table shapes!")

            for i in range(flag.shape[0]):
                flag[i, :] = flagbackup[i, :]


def main():
    parser = argparse.ArgumentParser(
        description="Restore flags from backup file.",
        prog="flagrestore.py",
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
    flagrestore(args.datafile, args.backupfile)


if __name__ == "__main__":
    main()
