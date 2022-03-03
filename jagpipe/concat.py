"""
concat.py
Concatenate multiple JAG SDHDF files. This expects "filled" SDHDF
files, like those produced by combine.py. Scans are merged across
files if necessary.

Copyright(C) 2021-2022 by
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
Trey Wenger - February 2022
"""

import os
import argparse
import shutil
import h5py

from . import __version__
from .utils import add_history


def concat(
    datafiles, outfile, verbose=False,
):
    """
    Concatenate multiple SDHDF files, and merge scans along the file breaks if
    necessary.

    Inputs:
        datafiles :: list of strings
            Input SDHDF datafiles. Should be supplied in time-ascending order
        outfile :: string
            Output SDHDF datafile
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    if os.path.exists(outfile):
        raise FileExistsError(f"Will not overwrite {outfile}")

    # sort datafiles
    datafiles.sort()

    # copy first data file to output file
    if verbose:
        print(f"Copying {datafiles[0]} to {outfile}")
    shutil.copy2(datafiles[0], outfile)

    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(outfile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        # add history items
        add_history(sdhdf, f"JAG-PIPELINE-CONCAT VERSION: {__version__}")
        for datafile in datafiles:
            add_history(sdhdf, f"JAG-PIPELINE-CONCAT DATAFILE: {datafile}")

        # get last scan number
        outf_scans = [
            key for key in sdhdf["data"]["beam_0"]["band_SB0"].keys() if "scan_" in key
        ]
        outf_scans = sorted(outf_scans, key=lambda scan_id: int(scan_id[5:]))
        last_scan = int(outf_scans[-1][5:])

        # Loop over input files
        for datafile in datafiles[1:]:
            if verbose:
                print(f"Adding {datafile}")

            with h5py.File(datafile, "r") as inf:
                # get input file scans
                inf_scans = [
                    key
                    for key in inf["data"]["beam_0"]["band_SB0"].keys()
                    if "scan_" in key
                ]
                inf_scans = sorted(inf_scans, key=lambda scan_id: int(scan_id[5:]))

                # Check if first scan of input file has same target of last
                # scan in outfile
                last_source = sdhdf[f"data/beam_0/band_SB0/scan_{last_scan}"].attrs[
                    "SOURCE"
                ]
                input_source = inf["data/beam_0/band_SB0/scan_0"].attrs["SOURCE"]

                if input_source == last_source:
                    if verbose:
                        print(f"Merging scans on {input_source}")

                    # merge scans
                    outf_scan = sdhdf[f"data/beam_0/band_SB0/scan_{last_scan}"]
                    inf_scan = inf["data/beam_0/band_SB0/scan_0"]
                    size = outf_scan["metadata"].shape[0]

                    # merge metadata
                    outf_scan["metadata"].resize(
                        size + inf_scan["metadata"].shape[0], axis=0,
                    )
                    outf_scan["metadata"][size:] = inf_scan["metadata"]

                    # merge flag
                    outf_scan["flag"].resize(
                        size + inf_scan["flag"].shape[0], axis=0,
                    )
                    outf_scan["flag"][size:] = inf_scan["flag"]

                    # merge data
                    outf_scan["data"].resize(
                        size + inf_scan["data"].shape[0], axis=0,
                    )
                    outf_scan["data"][size:] = inf_scan["data"]

                    # remove first scan from input file list
                    inf_scans = inf_scans[1:]

                # add input file scans to outfile
                for inf_scan in inf_scans:
                    if verbose:
                        source = inf[f"data/beam_0/band_SB0/{inf_scan}"].attrs["SOURCE"]
                        print(f"Copying scan on {source}")
                    last_scan += 1
                    inf.copy(
                        inf[f"data/beam_0/band_SB0/{inf_scan}"],
                        sdhdf["data/beam_0/band_SB0"],
                        f"scan_{last_scan}",
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple SDHDF datafiles",
        prog="concat.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "outfile", type=str, help="Output SDHDF file",
    )
    parser.add_argument(
        "datafiles", type=str, nargs="+", help="Input SDHDF file(s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose information",
    )
    args = parser.parse_args()
    concat(
        args.datafiles, args.outfile, verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
