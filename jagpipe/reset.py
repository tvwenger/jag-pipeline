"""
reset.py
Reset the cal and flag tables of a SDHDF data file.

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
import numpy as np

from . import __version__
from .utils import add_history


def reset(datafile):
    """
    Reset the cal state and flag tables of a SDHDF data file to False.

    Inputs:
        datafile :: string
            SDHDF filename

    Returns: Nothing
    """
    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        # add history items
        add_history(sdhdf, f"JAG-PIPELINE-RESET VERSION: {__version__}")

        # Loop over scans
        scans = [
            key for key in sdhdf["data"]["beam_0"]["band_SB0"].keys() if "scan" in key
        ]
        for scan in scans:
            sdhdf["data"]["beam_0"]["band_SB0"][scan]["flag"][:] = False
            metadata = np.copy(sdhdf["data"]["beam_0"]["band_SB0"][scan]["metadata"])
            metadata["CAL"][:] = False
            sdhdf["data"]["beam_0"]["band_SB0"][scan]["metadata"][:] = metadata


def main():
    parser = argparse.ArgumentParser(
        description="Reset SDHDF cal and flag tables",
        prog="reset.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    args = parser.parse_args()
    reset(args.datafile)


if __name__ == "__main__":
    main()
