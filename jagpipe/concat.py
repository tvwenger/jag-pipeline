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
import h5py
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import astropy.constants as c

from . import __version__
from .utils import add_history


def concat(
    datafiles, outfile, verbose=False,
):
    """
    Concatenate multiple SDHDF files, and merge scans along the file breaks if necessary.

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

    # Get source information
    sources = np.genfromtxt(sourcefile, dtype=None, names=True, encoding="utf-8")
    source_coords = SkyCoord(
        sources["RA"], sources["Dec"], frame="icrs", unit="hourangle,deg",
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
        time_buffer = None
        position_buffer = None

        # Loop over input data files
        for datai, datafile in enumerate(datafiles):
            if verbose:
                print(f"Reading {datafile}")
            with h5py.File(datafile, "r") as inf:
                # concatenate data long frequency axis
                raw_data = np.concatenate(
                    [
                        inf["data"]["beam_0"][band]["scan_0"]["data"][()]
                        for band in inf["data"]["beam_0"].keys()
                    ],
                    axis=-1,
                )

                # Read metadata
                raw_time = inf["data"]["beam_0"]["band_SB4"]["scan_0"]["time"][()]
                raw_position = inf["data"]["beam_0"]["band_SB4"]["scan_0"]["position"][
                    ()
                ]

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

            # add data to buffer
            if data_buffer is None:
                data_buffer = raw_data.copy()
                time_buffer = raw_time.copy()
                position_buffer = raw_position.copy()
            else:
                data_buffer = np.concatenate([data_buffer, raw_data], axis=0)
                time_buffer = np.concatenate([time_buffer, raw_time], axis=0)
                position_buffer = np.concatenate(
                    [position_buffer, raw_position], axis=0
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
                utc = Time(
                    time_buffer[0], format="isot", scale="utc", location=location
                )
                if verbose:
                    print(f"Processing {utc.isot}")
                position = position_buffer[0]
                data_buffer = np.delete(data_buffer, np.s_[0:timebin], axis=0)
                time_buffer = np.delete(time_buffer, np.s_[0:timebin], axis=0)
                position_buffer = np.delete(position_buffer, np.s_[0:timebin], axis=0)

                # Get position in ICRS
                coord = SkyCoord(
                    position[1],
                    position[0],
                    unit="deg",
                    frame="altaz",
                    location=location,
                    obstime=utc,
                ).icrs

                # get separation from each source
                seps = coord.separation(source_coords).to("rad")

                # check off source
                if seps.min() > hpbwfrac * hpbw:
                    continue
                idx = np.argmin(seps)

                # Check if we need to start a new scan
                if scan is None or scan.attrs["SOURCE"] != sources[idx]["Name"]:
                    # Flag last integration of previous scan, since the
                    # telescope likely started slewing during it
                    if scan is not None:
                        scan["flag"][-1] = flag_mask

                    # Create new scan
                    scan = init_scan(
                        band_sb0,
                        scan_count,
                        str(sources[idx]["Name"]),
                        source_coords[idx],
                        len(freqaxis),
                        chunks=chunks,
                    )
                    scan_count += 1

                # Add metadata row
                hourangle = utc.sidereal_time("mean").hourangle - coord.ra.hourangle
                dat = np.array(
                    (
                        utc.mjd,
                        utc.isot,
                        position[1],
                        position[0],
                        hourangle,
                        coord.dec.deg,
                        0,
                    ),
                    dtype=scan["metadata"].dtype,
                )
                scan["metadata"].resize(scan["metadata"].shape[0] + 1, axis=0)
                scan["metadata"][-1] = dat

                # store flags
                scan["flag"].resize(scan["flag"].shape[0] + 1, axis=0)
                scan["flag"][-1] = null_mask

                # store data
                scan["data"].resize(scan["data"].shape[0] + 1, axis=0)
                scan["data"][-1] = data

        # Flag last integration of previous scan, since the
        # telescope likely started slewing during it
        if scan is not None:
            scan["flag"][-1] = flag_mask


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
    combine(
        args.datafiles, args.outfile, verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
