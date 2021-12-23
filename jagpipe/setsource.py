"""
setsource.py
Set source name in SDHDF position dataset based on position of telescope
and a database of source coordinates. Flag off-source integrations.

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
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c

from . import __version__


class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    argparse formatter wrapper that supports line breaks in help text
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return argparse.ArgumentDefaultsHelpFormatter._split_lines(self, text, width)


def setsource(
    datafile, sourcefile, hpbwfrac=0.1,
):
    """
    Set source names and flag off-source integrations.

    sourcefile has the following format:
    Name     RA           Dec
    #        J2000        J2000
    #        hh:mm:ss.sss +dd:mm:ss.sss
    CYGA     19:59:28.320 +40:44:01.720
    CYGA_OFF 20:30:28.320 +40:44:01.720

    Inputs:
        datafile :: string
            SDHDF file
        sourcefile :: string
            ASCII file containing source names and J2000 positions
        hpbwfrac :: scalar
            Telescope position must be within this fraction of the
            HPBW at the highest frequency to be considered a match
            to a source position.

    Returns: Nothing
    """
    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        # initialize
        sdhdf["metadata"].attrs["JAG-PIPELINE-SETSOURCE-VERSION"] = __version__
        sdhdf["metadata"].attrs["JAG-PIPELINE-SETSOURCE-EXECTIME"] = Time.now().isot
        sdhdf["metadata"].attrs["JAG-PIPELINE-SETSOURCE-SOURCEFILE"] = sourcefile

        # get data, mask, and positions
        frequency = sdhdf["data"]["beam_0"]["band_SB0"]["frequency"]
        freq_unit = sdhdf["data"]["beam_0"]["band_SB0"]["frequency"].attrs["UNIT"]
        flag = sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["flag"]
        position = np.copy(sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["position"])

        # JAG HPBW
        hpbw = 1.2 * c.c / (np.max(frequency) * u.Unit(freq_unit) * 26.0 * u.m)
        hpbw = hpbw.to("").value * u.radian

        # Coordinates at start of integration
        coords = SkyCoord(position["RA"], position["DEC"], unit="deg", frame="icrs")

        # Get source information
        sources = np.genfromtxt(sourcefile, dtype=None, names=True, encoding="utf-8")
        source_coords = SkyCoord(
            sources["RA"], sources["Dec"], frame="icrs", unit="hourangle,deg",
        )

        # Loop over integrations
        for i, coord in enumerate(coords):
            # get separation from each source
            seps = coord.separation(source_coords).to("rad")

            # check off source
            if seps.min() > hpbwfrac * hpbw:
                position[i]["SOURCE"] = "N/A"
                flag[i] = True
            else:
                idx = np.argmin(seps)
                position[i]["SOURCE"] = sources[idx]["Name"]

        # Loop over integrations and flag last integration per
        # source, since the telescope probably moved during that
        # integration.
        for i in range(len(position)):
            if i + 1 < len(position):
                if position[i]["SOURCE"] != position[i + 1]["SOURCE"]:
                    flag[i] = True

        # save position
        sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["position"][:] = position


def main():
    parser = argparse.ArgumentParser(
        description="Assign source names based on telescope position, flag off-source",
        prog="setsource.py",
        formatter_class=SmartFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    parser.add_argument(
        "sourcefile",
        type=str,
        help=(
            "R|ASCII file containing source names and J2000 positions\n"
            + "Format like:\n"
            + "Name     RA           Dec\n"
            + "#        J2000        J2000\n"
            + "#        hh:mm:ss.sss +dd:mm:ss.sss\n"
            + "CYGA     19:59:28.320 +40:44:01.720\n"
            + "CYGA_OFF 20:30:28.320 +40:44:01.720"
        ),
    )
    parser.add_argument(
        "--hpbwfrac",
        type=float,
        default=0.1,
        help="Maximum offset relative to HPBW to match source position",
    )
    args = parser.parse_args()
    setsource(
        args.datafile, args.sourcefile, hpbwfrac=args.hpbwfrac,
    )


if __name__ == "__main__":
    main()
