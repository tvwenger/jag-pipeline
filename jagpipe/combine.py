"""
combine.py
Read multiple JAG SDHDF files, optionally bin in time and frequency,
assign sources based on telescope position, add "cal" and "flag"
tables, remove off-source integrations, break observation into
scans based on position, and output a single SDHDF file.

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
Trey Wenger - v0.5a1 - January 2023
    Backwards compatibility with old filler versions.
"""

import os
import argparse
import h5py
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import astropy.constants as c

from . import __version__
from .utils import add_history


class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    argparse formatter wrapper that supports line breaks in help text
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return argparse.ArgumentDefaultsHelpFormatter._split_lines(self, text, width)


def init_scan(band, scan_num, source, source_coord, num_freq, chunks=None):
    """
    Add a new scan to a SDHDF "band" group.

    Inputs:
        band :: h5py.Group
            SDHDF "band" group
        scan_num :: integer
            Scan number
        source :: string
            Source name
        source_coord :: astropy.Coordinate
            Source coordinate
        num_freq :: integer
            Number of frequency channels
        chunks :: tuple of integers
            Chunk shape

    Returns :: scan
        scan :: h5py.Group
            New scan group
    """
    scan = band.create_group(f"scan_{scan_num}")
    scan.attrs["NAME"] = f"scan_{scan_num}"
    scan.attrs["DESCRIPTION"] = f"Data specific to scan {scan_num}"
    if "OFF" in source:
        scan.attrs["TYPE"] = "off"
    else:
        scan.attrs["TYPE"] = "on"
    scan.attrs["SOURCE"] = source
    scan.attrs["RA"] = source_coord.ra.deg
    scan.attrs["DEC"] = source_coord.dec.deg

    # Scan metadata
    dat = np.array(
        [],
        dtype=[
            ("MJD", "f8"),
            ("UTC", "S23"),
            ("AZIMUTH", "f8"),
            ("ELEVATION", "f8"),
            ("HOURANGLE", "f8"),
            ("DECLINATION", "f8"),
            ("CAL", "?"),
        ],
    )
    scan_metadata = scan.create_dataset("metadata", data=dat, maxshape=(None,))
    scan_metadata.attrs["NAME"] = "metadata"
    scan_metadata.attrs[
        "DESCRIPTION"
    ] = "Observation metadata at start of each integration"
    scan_metadata.attrs["COLUMN_FRAMES"] = ["", "", "", "", "", "J2000"]
    scan_metadata.attrs["COLUMN_UNITS"] = [
        "",
        "",
        "deg",
        "deg",
        "hours",
        "deg",
    ]

    # Scan data
    dat = np.empty((0, 4, num_freq), dtype=float)
    scan_data = scan.create_dataset(
        "data",
        data=dat,
        maxshape=(None, 4, num_freq),
        chunks=chunks,
    )
    scan_data.attrs["NAME"] = "data"
    scan_data.attrs["DESCRIPTION"] = "Data"
    scan_data.attrs["DIMENSION_LABELS"] = [
        "time",
        "polarization",
        "channel",
    ]
    scan_data.attrs["TYPE"] = "Relative Power"
    scan_data.attrs["UNIT"] = ""

    # Flag storage
    data = np.empty((0, num_freq), dtype=bool)

    scan_flag = scan.create_dataset(
        "flag",
        data=data,
        maxshape=(None, num_freq),
        chunks=(chunks[0], chunks[2]),
    )
    scan_flag.attrs["NAME"] = "flag"
    scan_flag.attrs["DESCRIPTION"] = "Flag"
    scan_flag.attrs["DIMENSION_LABELS"] = ["time", "channel"]
    return scan


def combine(
    datafiles,
    outfile,
    exposure=None,
    chanbin=1,
    timebin=1,
    sourcefile=None,
    hpbwfrac=0.1,
    minint=5,
    verbose=False,
):
    """
    Read data files, apply binning, set source information, split scans, save to one
    output file.

    sourcefile has the following format:
    Name     RA           Dec
    #        J2000        J2000
    #        hh:mm:ss.sss +dd:mm:ss.sss
    CYGA     19:59:28.320 +40:44:01.720
    CYGA_OFF 20:30:28.320 +40:44:01.720

    Inputs:
        datafiles :: list of strings
            Input SDHDF datafiles
        outfile :: string
            Output SDHDF datafile
        exposure :: float (seconds)
            Overwrite exposure to this value
        chanbin :: integer
            Number of channels to bin
        timebin :: integer
            Number of time intervals to bin
        sourcefile :: string
            ASCII file containing source names and J2000 positions
        hpbwfrac :: scalar
            Telescope position must be within this fraction of the
            HPBW at the highest frequency to be considered a match
            to a source position.
        minint :: integer
            Minimum number of integrations to include in a scan. Useful
            to exclude integrations that are just slews through a position
            in sourcefile.
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    if os.path.exists(outfile):
        raise FileExistsError(f"Will not overwrite {outfile}")

    # sort datafiles
    datafiles.sort()

    # Get source information
    if sourcefile is not None:
        sources = np.genfromtxt(sourcefile, dtype=None, names=True, encoding="utf-8")
        source_coords = SkyCoord(
            sources["RA"],
            sources["Dec"],
            frame="icrs",
            unit="hourangle,deg",
        )

    # DRAO position
    location = EarthLocation.from_geodetic(
        240.3810197222, 49.3210230556, height=546.566
    )

    # Read first datafile to get frequency axis and exposure time
    # Assume these do not change for the duratin of the observation
    with h5py.File(datafiles[0], "r") as inf:
        if exposure is None:
            exposure = timebin * inf["data"]["beam_0"]["band_SB4"].attrs["EXPOSURE"]
        freqaxis = np.concatenate(
            [
                inf["data"]["beam_0"][band]["frequency"][()]
                for band in inf["data"]["beam_0"].keys()
            ]
        )
        # bin
        freqaxis = np.array(
            [
                freqaxis[i : i + chanbin].mean()
                for i in range(0, freqaxis.shape[0], chanbin)
            ]
        )

    # JAG HPBW
    hpbw = 1.2 * c.c / (np.max(freqaxis) * u.Hz * 26.0 * u.m)
    hpbw = hpbw.to("").value * u.radian

    # Chunk shape for data storage
    # For float, this is about 12 MB
    chunk_time = 20
    chunk_pol = 4
    chunk_freq = min(20000, len(freqaxis))
    chunks = (chunk_time, chunk_pol, chunk_freq)

    with h5py.File(outfile, "w") as sdhdf:
        # Setup metadata structure
        metadata = sdhdf.create_group("metadata")
        metadata.attrs["NAME"] = "metadata"
        metadata.attrs["DESCRIPTION"] = "Observation metadata"
        metadata.attrs["SDHDF_VERSION"] = "v2.0-alpha20210923"
        metadata.attrs["OBSERVER"] = "Wenger"
        metadata.attrs["TELESCOPE"] = "DRAO JAG 26m"
        metadata.attrs["PROJECTID"] = "DRAO-JAG-21B-000"
        metadata.attrs["RECEIVER"] = "CHIME"
        metadata.attrs["INSTRUMENT"] = "Backend Name"
        metadata.attrs["DATE"] = Time.now().isot

        # History
        dat = np.array([], dtype=[("UTC", "S23"), ("MESSAGE", "S256")])
        metadata.create_dataset("history", data=dat, maxshape=(None,))

        # Data group
        data = sdhdf.create_group("data")
        data.attrs["NAME"] = "data"
        data.attrs["DESCRIPTION"] = "Observation data"

        # Data - Beam 0
        beam_0 = data.create_group("beam_0")
        beam_0.attrs["NAME"] = "beam_0"
        beam_0.attrs["DESCRIPTION"] = "Data specific to antenna beam 0"

        # Data - Beam 0 - Band SB0
        band_sb0 = beam_0.create_group("band_SB0")
        band_sb0.attrs["NAME"] = "band_SB0"
        band_sb0.attrs["DESCRIPTION"] = "Data specific to frequency band SB0"
        band_sb0.attrs["EXPOSURE"] = exposure

        # Data - Beam 0 - Band SB0 - Frequency
        frequency = band_sb0.create_dataset("frequency", data=freqaxis)
        frequency.attrs["NAME"] = "frequency"
        frequency.attrs["DESCRIPTION"] = "Channel center frequencies"
        frequency.attrs["FRAME"] = "Topocentric"
        frequency.attrs["UNIT"] = "Hz"

        # Data - Beam 0 - Band SB0 - Polarization
        polarization_data = np.array(["XX", "YY", "Re(XY)", "Im(XY)"], dtype="S")
        polarization = band_sb0.create_dataset("polarization", data=polarization_data)
        polarization.attrs["NAME"] = "polarization"
        polarization.attrs["DESCRIPTION"] = "Polarization types"

        # add history items
        add_history(sdhdf, f"JAG-PIPELINE-COMBINE VERSION: {__version__}")
        for datafile in datafiles:
            add_history(sdhdf, f"JAG-PIPELINE-COMBINE DATAFILE: {datafile}")
        add_history(sdhdf, f"JAG-PIPELINE-COMBINE TIMEBIN: {timebin}")
        add_history(sdhdf, f"JAG-PIPELINE-COMBINE CHANBIN: {chanbin}")
        add_history(sdhdf, f"JAG-PIPELINE-COMBINE SOURCEFILE: {sourcefile}")
        add_history(sdhdf, f"JAG-PIPELINE-COMBINE HPBWFRAC: {hpbwfrac}")

        scan = None
        scan_count = 0

        # storage for binned data and metadata
        data_buffer = None
        metadata_buffer = None
        position_buffer = None
        time_buffer = None

        # tracking off source integrations
        num_off_source = 0
        first_off_source = None
        last_off_source = None

        # Loop over input data files
        for datai, datafile in enumerate(datafiles):
            with h5py.File(datafile, "r") as inf:
                # concatenate data long frequency axis
                raw_data = np.concatenate(
                    [
                        inf["data"]["beam_0"][band]["scan_0"]["data"][()]
                        for band in inf["data"]["beam_0"].keys()
                    ],
                    axis=-1,
                )

                # flag for old metadata format
                old_format = (
                    "metadata" not in inf["data"]["beam_0"]["band_SB4"]["scan_0"].keys()
                )

                # Read metadata, backwards compatible
                if old_format:
                    raw_time = inf["data"]["beam_0"]["band_SB4"]["scan_0"]["time"][()]
                    raw_position = inf["data"]["beam_0"]["band_SB4"]["scan_0"][
                        "position"
                    ][()]
                else:
                    raw_metadata = inf["data"]["beam_0"]["band_SB4"]["scan_0"][
                        "metadata"
                    ][()]

            # bin data in frequency
            pad = raw_data.shape[2] % chanbin
            if pad != 0:
                pad = chanbin - pad
            raw_data = np.pad(
                raw_data,
                ((0, 0), (0, 0), (0, pad)),
                mode="constant",
                constant_values=np.nan,
            )
            raw_data = np.mean(raw_data.reshape(-1, chanbin), axis=1).reshape(
                raw_data.shape[0], raw_data.shape[1], -1
            )

            # add data and metadata to buffer
            if data_buffer is None:
                data_buffer = raw_data.copy()
                if old_format:
                    time_buffer = raw_time.copy()
                    position_buffer = raw_position.copy()
                else:
                    metadata_buffer = raw_metadata.copy()
            else:
                data_buffer = np.concatenate([data_buffer, raw_data], axis=0)
                if old_format:
                    time_buffer = np.concatenate([time_buffer, raw_time], axis=0)
                    position_buffer = np.concatenate(
                        [position_buffer, raw_position], axis=0
                    )
                else:
                    metadata_buffer = np.concatenate(
                        [metadata_buffer, raw_metadata], axis=0
                    )

            # flag and null flag masks
            flag_mask = np.ones_like(freqaxis.shape[0], dtype=bool)
            null_mask = np.zeros_like(freqaxis.shape[0], dtype=bool)

            while True:
                # if there aren't enough time rows in the buffer, and if there
                # is another datafile, then load the next data file
                if data_buffer.shape[0] < timebin and datai < len(datafiles) - 1:
                    break

                # if we're out of rows, then we're done
                if data_buffer.shape[0] == 0:
                    break

                # extract rows from buffer, bin in time
                data = np.mean(data_buffer[0:timebin], axis=0)
                if old_format:
                    utc = Time(
                        time_buffer[0], format="isot", scale="utc", location=location
                    )
                else:
                    utc = Time(
                        metadata_buffer["utc"][0],
                        format="isot",
                        scale="utc",
                        location=location,
                    )
                if verbose:
                    print(f"Datafile: {datafile} Integration: {utc.isot}", end="\r")
                if old_format:
                    azimuth = position_buffer[0][1]
                    elevation = position_buffer[0][0]
                else:
                    elevation = metadata_buffer["elevation"][0]
                    azimuth = metadata_buffer["azimuth"][0]

                corrupted = False
                noise_state = False
                if (
                    not old_format
                    and "corrupted" in metadata_buffer.dtype.names
                    and np.any(metadata_buffer["corrupted"][0:timebin])
                ):
                    corrupted = True
                if not old_format and "noise_state" in metadata_buffer.dtype.names:
                    if np.any(
                        metadata_buffer["noise_state"][0:timebin] == 1
                    ) and np.any(metadata_buffer["noise_state"][0:timebin] == 0):
                        # cal state changed during integration
                        corrupted = True
                        noise_state = True
                    elif np.any(metadata_buffer["noise_state"][0:timebin] == 1):
                        noise_state = True

                data_buffer = np.delete(data_buffer, np.s_[0:timebin], axis=0)
                if old_format:
                    time_buffer = np.delete(time_buffer, np.s_[0:timebin], axis=0)
                    position_buffer = np.delete(
                        position_buffer, np.s_[0:timebin], axis=0
                    )
                else:
                    metadata_buffer = np.delete(
                        metadata_buffer, np.s_[0:timebin], axis=0
                    )

                # Get position in ICRS
                coord = SkyCoord(
                    azimuth,
                    elevation,
                    unit="deg",
                    frame="altaz",
                    location=location,
                    obstime=utc,
                ).icrs

                # get separation from each source
                if sourcefile is not None:
                    seps = coord.separation(source_coords).to("rad")

                    # check off source
                    if seps.min() > hpbwfrac * hpbw:
                        num_off_source += 1
                        if first_off_source is None:
                            print()
                            print("Telescope is off-source")
                            print(f"Position (J2000): {coord.icrs.to_string('hmsdms', sep=':')}")
                            first_off_source = utc.isot
                        elif num_off_source % 100 == 0:
                            print()
                            print("Telescope remains off-source")
                            print(f"Position (J2000): {coord.icrs.to_string('hmsdms', sep=':')}")
                        last_off_source = utc.isot
                        continue
                    elif num_off_source > 0:
                        print()
                        print(f"Off-source integrations: {num_off_source}")
                        print(f"From {first_off_source}")
                        print(f"To {last_off_source}")
                    idx = np.argmin(seps)

                    num_off_source = 0
                    first_off_source = None
                    last_off_source = None

                # Check if we need to start a new scan
                if scan is None or (
                    sourcefile is not None
                    and scan.attrs["SOURCE"] != sources[idx]["Name"]
                ):
                    # Flag last integration of previous scan, since the
                    # telescope likely started slewing during it
                    if scan is not None:
                        scan["flag"][-1] = flag_mask

                    # delete last scan if it did not meet minint threshold
                    if scan is not None and scan["data"].shape[0] < minint:
                        if verbose:
                            print()
                            print(
                                f"Deleting {scan.attrs['NAME']} "
                                + f"({scan.attrs['SOURCE']}) "
                                + f"which only has {scan['data'].shape[0]} integrations"
                            )
                        del sdhdf["data"]["beam_0"]["band_SB0"][scan.attrs["NAME"]]
                        scan_count -= 1

                    # Create new scan
                    if sourcefile is None:
                        scan = init_scan(
                            band_sb0,
                            scan_count,
                            "UNKNOWN",
                            SkyCoord(0.0, 0.0, frame="icrs", unit="hourangle,deg"),
                            len(freqaxis),
                            chunks=chunks,
                        )
                    else:
                        scan = init_scan(
                            band_sb0,
                            scan_count,
                            str(sources[idx]["Name"]),
                            source_coords[idx],
                            len(freqaxis),
                            chunks=chunks,
                        )
                    scan_count += 1
                    if verbose:
                        print()
                        print(
                            f"Creating scan {scan.attrs['NAME']} "
                            + f"({scan.attrs['SOURCE']})"
                        )

                # Add metadata row
                hourangle = utc.sidereal_time("mean").hourangle - coord.ra.hourangle
                dat = np.array(
                    (
                        utc.mjd,
                        utc.isot,
                        azimuth,
                        elevation,
                        hourangle,
                        coord.dec.deg,
                        1 if noise_state else 0,
                    ),
                    dtype=scan["metadata"].dtype,
                )
                scan["metadata"].resize(scan["metadata"].shape[0] + 1, axis=0)
                scan["metadata"][-1] = dat

                # store flags
                scan["flag"].resize(scan["flag"].shape[0] + 1, axis=0)
                scan["flag"][-1] = flag_mask if corrupted else null_mask

                # store data
                scan["data"].resize(scan["data"].shape[0] + 1, axis=0)
                scan["data"][-1] = data

        # Flag last integration of previous scan, since the
        # telescope likely started slewing during it
        if scan is not None:
            scan["flag"][-1] = flag_mask


def main():
    parser = argparse.ArgumentParser(
        description="Combine and bin JAG datafiles",
        prog="combine.py",
        formatter_class=SmartFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output SDHDF file",
    )
    parser.add_argument(
        "datafiles",
        type=str,
        nargs="+",
        help="Input SDHDF file(s)",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Overwrite output exposure to this value",
    )
    parser.add_argument(
        "-c",
        "--chanbin",
        type=int,
        default=1,
        help="Channel bin size",
    )
    parser.add_argument(
        "-t",
        "--timebin",
        type=int,
        default=1,
        help="Time bin size",
    )
    parser.add_argument(
        "-s",
        "--sourcefile",
        type=str,
        default=None,
        help=(
            "R|ASCII file containing source names and J2000 positions\n"
            + "If not supplied, all data are concatenated into one scan\n"
            + "Format like:\n"
            + "Name     RA           Dec\n"
            + "#        J2000        J2000\n"
            + "#        hh:mm:ss.sss +dd:mm:ss.sss\n"
            + "CYGA     19:59:28.320 +40:44:01.720\n"
            + "CYGA_OFF 20:30:28.320 +40:44:01.720"
        ),
    )
    parser.add_argument(
        "-f",
        "--hpbwfrac",
        type=float,
        default=0.1,
        help="Maximum offset relative to HPBW to match source position",
    )
    parser.add_argument(
        "--minint",
        type=int,
        default=5,
        help="Mininmum number of integrations required to create a scan",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose information",
    )
    args = parser.parse_args()
    combine(
        args.datafiles,
        args.outfile,
        exposure=args.exposure,
        chanbin=args.chanbin,
        timebin=args.timebin,
        sourcefile=args.sourcefile,
        hpbwfrac=args.hpbwfrac,
        minint=args.minint,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
