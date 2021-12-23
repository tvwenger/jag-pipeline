"""
combine.py
Read multiple JAG SDHDF files, optionally bin in time and frequency,
and the output a single SDHDF file with an additional "flag" dataset.

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
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from . import __version__
from .init_sdhdf import init_sdhdf


def combine(
    datafiles, outfile, chanbin=1, timebin=1, verbose=False,
):
    """
    Read data files, apply binning, save to one output file.

    Inputs:
        datafiles :: list of strings
            The input datafiles
        outfile :: string
            The output datafile
        chanbin :: integer
            Number of channels to bin
        timebin :: integer
            Number of time intervals to bin
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    if os.path.exists(outfile):
        raise FileExistsError(f"Will not overwrite: {outfile}")

    # sort datafiles
    datafiles.sort()

    # DRAO position
    location = EarthLocation.from_geodetic(
        240.3810197222, 49.3210230556, height=546.566
    )

    # Read first datafile to get frequency axis and exposure time
    # Assume these do not change for the duratin of the observation
    with h5py.File(datafiles[0], "r") as inf:
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

    with h5py.File(outfile, "w") as outf:
        # initialize
        init_sdhdf(outf, freqaxis, "Topocentric", exposure)
        metadata = outf["metadata"]
        scan_position = outf["data"]["beam_0"]["band_SB0"]["scan_0"]["position"]
        scan_flag = outf["data"]["beam_0"]["band_SB0"]["scan_0"]["flag"]
        scan_cal = outf["data"]["beam_0"]["band_SB0"]["scan_0"]["cal"]
        scan_data = outf["data"]["beam_0"]["band_SB0"]["scan_0"]["data"]
        metadata.attrs["JAG-PIPELINE-COMBINE-VERSION"] = __version__
        metadata.attrs["JAG-PIPELINE-COMBINE-EXECTIME"] = Time.now().isot
        metadata.attrs["JAG-PIPELINE-COMBINE-DATAFILES[0]"] = datafiles[0]
        metadata.attrs["JAG-PIPELINE-COMBINE-TIMEBIN"] = timebin
        metadata.attrs["JAG-PIPELINE-COMBINE-CHANBIN"] = chanbin

        # process
        data_buffer = None
        time_buffer = None
        position_buffer = None
        for datai, datafile in enumerate(datafiles):
            if verbose:
                print(f"Reading {datafile}")
            with h5py.File(datafile, "r") as inf:
                # concatenate frequency axis
                raw_data = np.concatenate(
                    [
                        inf["data"]["beam_0"][band]["scan_0"]["data"][()]
                        for band in inf["data"]["beam_0"].keys()
                    ],
                    axis=-1,
                )

                raw_time = inf["data"]["beam_0"]["band_SB4"]["scan_0"]["time"][()]
                raw_position = inf["data"]["beam_0"]["band_SB4"]["scan_0"]["position"][
                    ()
                ]

            # bin in frequency
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

            # null mask
            mask = np.zeros_like(freqaxis.shape[0], dtype=bool)

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
                utc = Time(time_buffer[0], format="isot", scale="utc")
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

                # store position
                position_row = np.array(
                    (
                        "",
                        utc.isot,
                        utc.mjd,
                        position[1],
                        position[0],
                        coord.ra.deg,
                        coord.dec.deg,
                    ),
                    dtype=scan_position.dtype,
                )
                scan_position.resize(scan_position.shape[0] + 1, axis=0)
                scan_position[-1] = position_row

                # store flags
                scan_flag.resize(scan_flag.shape[0] + 1, axis=0)
                scan_flag[-1] = mask

                # store null call
                scan_cal.resize(scan_cal.shape[0] + 1, axis=0)
                scan_cal[-1] = False

                # store data
                scan_data.resize(scan_data.shape[0] + 1, axis=0)
                scan_data[-1] = data


def main():
    parser = argparse.ArgumentParser(
        description="Combine and bin JAG datafiles",
        prog="combine.py",
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
        "-c", "--chanbin", type=int, default=1, help="Channel bin size",
    )
    parser.add_argument(
        "-t", "--timebin", type=int, default=1, help="Time bin size",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose information",
    )
    args = parser.parse_args()
    combine(
        args.datafiles,
        args.outfile,
        chanbin=args.chanbin,
        timebin=args.timebin,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
