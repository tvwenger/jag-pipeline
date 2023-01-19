"""
listobs.py
Print SDHDF observation information (frequency setup, scan
information, etc.)

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
Trey Wenger - March 2022
"""

import os
import argparse
import h5py

from . import __version__


def listobs(datafile, listfile=None):
    """
    Read SDHDF file, print observation information

    Inputs:
        datafile :: string
            SDHDF datafile
        listfile :: string
            If not None, also output information to this file

    Returns: Nothing
    """
    if listfile is not None and os.path.exists(listfile):
        raise FileExistsError(f"Will not overwrite {listfile}")
    output = []

    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r", rdcc_nbytes=cache_size) as sdhdf:
        output += [f"jagpipe-listobs: {datafile}"]
        output += []
        output += ["==========================================================="]
        output += []

        # get metadata
        output += ["SDHDF Metadata"]
        for key in sdhdf["metadata"].attrs.keys():
            if key in ["NAME", "DESCRIPTION"]:
                continue
            output += [f"{key}: {sdhdf['metadata'].attrs[key]}"]
        output += []
        output += ["==========================================================="]
        output += []

        # get history
        output += ["SDHDF History"]
        for row in sdhdf["metadata"]["history"][:]:
            output += [f"{row[0].decode('utf-8')} {row[1].decode('utf-8')}"]
        output += []
        output += ["==========================================================="]
        output += []

        # get number of beams
        beams = [key for key in sdhdf["data"].keys() if "beam_" in key]
        beams = sorted(beams, key=lambda beam_id: int(beam_id[5:]))
        output += [f"Number of beams: {len(beams)}"]
        output += ["Beams: " + "; ".join(beams)]
        output += []
        output += ["==========================================================="]
        output += []

        # Loop over beams
        for beam in beams:
            # get number of bands
            bands = [key for key in sdhdf["data"][beam].keys() if "band_SB" in key]
            bands = sorted(bands, key=lambda band_id: int(band_id[7:]))
            output += [f"Beam: {beam}"]
            output += [f"Number of bands: {len(bands)}"]
            output += ["Bands: " + "; ".join(bands)]
            output += []
            output += ["==========================================================="]
            output += []

            # Loop over bands
            for band in bands:
                # get number of scans
                scans = [
                    key for key in sdhdf["data"][beam][band].keys() if "scan_" in key
                ]
                scans = sorted(scans, key=lambda scan_id: int(scan_id[5:]))
                output += [f"Beam: {beam}; Band: {band}"]
                output += [
                    f"Exposure (s): {sdhdf['data'][beam][band].attrs['EXPOSURE']}"
                ]
                output += [f"Number of scans: {len(scans)}"]

                # get frequency axis information
                freqaxis = sdhdf["data"][beam][band]["frequency"]
                firstfreq = freqaxis[0] / 1e6
                freqdelta = (freqaxis[1] - freqaxis[0]) / 1e3
                nfreq = len(freqaxis)
                lastfreq = freqaxis[-1] / 1e6
                output += [f"Number of frequency channels: {nfreq}"]
                output += [f"First frequency channel (MHz): {firstfreq}"]
                output += [f"Last frequency channel (MHz): {lastfreq}"]
                output += [f"Channel width (kHz): {freqdelta}"]
                output += []

                # print scan information
                # scan_id   source          type   RA         Dec        first_int               last_int                num_int
                #                                  deg-J2000  deg-J2000  UTC                     UTC
                # scan_1234 G123.456+12.456 RASTER 123.456789 -15.123456 2022-01-13T04:22:04.000 2022-01-13T04:22:04.000 12345
                # 123456789 123456789012345 123456 1234567890 1234567890 12345678901234567890123 12345678901234567890123 1234567
                headerfmt = "{0:9} {1:15} {2:6} {3:10} {4:10} {5:23} {6:23} {7:7}"
                rowfmt = "{0:9} {1:15} {2:6} {3:10.6f} {4:+10.6f} {5:23} {6:23} {7:7}"
                output += [
                    headerfmt.format(
                        "scan_id",
                        "source",
                        "type",
                        "RA",
                        "Dec",
                        "first_int",
                        "last_int",
                        "num_int",
                    )
                ]
                for scan in scans:
                    output += [
                        rowfmt.format(
                            scan,
                            sdhdf["data"][beam][band][scan].attrs["SOURCE"],
                            sdhdf["data"][beam][band][scan].attrs["TYPE"],
                            sdhdf["data"][beam][band][scan].attrs["RA"],
                            sdhdf["data"][beam][band][scan].attrs["DEC"],
                            sdhdf["data"][beam][band][scan]["metadata"]["UTC"][
                                0
                            ].decode("utf-8"),
                            sdhdf["data"][beam][band][scan]["metadata"]["UTC"][
                                -1
                            ].decode("utf-8"),
                            sdhdf["data"][beam][band][scan]["metadata"].shape[0],
                        )
                    ]
                output += []
                output += [
                    "==========================================================="
                ]
                output += []
    # print
    if listfile is not None:
        with open(listfile, "w") as f:
            for out in output:
                f.write(out + "\n")
    else:
        for out in output:
            print(out)


def main():
    parser = argparse.ArgumentParser(
        description="Print SDHDF observation summary",
        prog="listobs.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="Input SDHDF file",
    )
    parser.add_argument(
        "--listfile", type=str, default=None, help="Write output to this file",
    )
    args = parser.parse_args()
    listobs(
        args.datafile, listfile=args.listfile,
    )


if __name__ == "__main__":
    main()
