"""
findcal.py
Identify calibration integrations and flag cal off->on
and on->off integrations.

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
from scipy.signal import argrelmax

from . import __version__
from .utils import add_history
from .flagchan import rolling_mean


def findcal(
    datafile, start=0, end=-1, duration=30.0, period=60.0, verbose=False,
):
    """
    Identify calibration integrations in SDHDF file and
    mask cal off->on and on->off integrations.

    Inputs:
        datafile :: string
            SDHDF file
        start, end :: integers
            Frequnecy channel boundaries over which to integrate
        duration :: scalar (seconds)
            Cal-on duration
        period :: scalar (seconds)
            Cal period
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        # add history items
        add_history(sdhdf, f"JAG-PIPELINE-FINDCAL VERSION: {__version__}")
        add_history(sdhdf, f"JAG-PIPELINE-FINDCAL START: {start}")
        add_history(sdhdf, f"JAG-PIPELINE-FINDCAL END: {end}")
        add_history(sdhdf, f"JAG-PIPELINE-FINDCAL DURATION: {duration}")
        add_history(sdhdf, f"JAG-PIPELINE-FINDCAL PERIOD: {period}")

        # Loop over scans
        scans = [
            key for key in sdhdf["data"]["beam_0"]["band_SB0"].keys() if "scan" in key
        ]
        scans = sorted(scans, key=lambda scan: int(scan[5:]))
        for scan in scans:
            if verbose:
                print(f"Processing {scan}")
            # get metadata, data, and mask
            metadata = np.copy(sdhdf["data"]["beam_0"]["band_SB0"][scan]["metadata"])
            data = sdhdf["data"]["beam_0"]["band_SB0"][scan]["data"]
            flag = sdhdf["data"]["beam_0"]["band_SB0"][scan]["flag"]
            exposure = sdhdf["data"]["beam_0"]["band_SB0"].attrs["EXPOSURE"]

            # get integration timestamps in seconds
            scantimes = metadata["MJD"] * 24.0 * 3600.0

            # Median over range of channels
            time_series = np.ones_like(scantimes) * np.nan
            if verbose:
                print("Reading data...")
            for i in range(data.shape[0]):
                print(i, end="\r")
                total_power = data[i, 0, start:end] + data[i, 1, start:end]
                total_power[flag[i, start:end]] = np.nan
                if not np.all(np.isnan(total_power)):
                    time_series[i] = np.nanmedian(total_power)

            # Interpolate onto uniform time axis with spacing exposure / 4
            start_time = scantimes[0]
            end_time = scantimes[-1] + exposure
            scan_duration = end_time - start_time
            plot_num_int = int(np.round(4.0 * scan_duration / exposure))
            interp_times = start_time + np.arange(plot_num_int) * exposure / 4.0
            interp_time_series = np.interp(interp_times, scantimes, time_series)

            # Rolling mean over cal duration (increase SNR)
            if verbose:
                print("Calculating statistics...")
            duration_window = int(4.0 * duration / exposure)
            if duration_window % 2 == 0:
                duration_window += 1
            duration_mean = rolling_mean(interp_time_series, duration_window)

            # Rolling mean over cal period (remove global variations)
            period_window = int(4.0 * period / exposure)
            if period_window % 2 == 0:
                period_window += 1
            period_mean = rolling_mean(interp_time_series, period_window)

            # Cal signal shape centered on peak
            tophat = np.zeros(period_window)
            st = period_window // 2 - duration_window // 2
            en = period_window // 2 + duration_window // 2 + 1
            tophat[st:en] = 1.0

            if verbose:
                print("Identifying cal integrations...")
            # catch where original data are all flagged
            dat = duration_mean - period_mean
            dat[np.isnan(interp_time_series)] = np.nan
            # convolve twice (better SNR), handling nans
            num_convolutions = 2
            for i in range(num_convolutions):
                isnan = np.isnan(dat)
                dat[isnan] = 0.0
                isnan_dat = np.ones(dat.shape)
                isnan_dat[isnan] = 0.0
                dat = np.convolve(dat, tophat, mode="same")
                isnan_conv = np.convolve(isnan_dat, tophat, mode="same")
                dat[isnan_conv / isnan_conv.max() < 0.5] = np.nan

            # identify peaks
            idx = argrelmax(dat)[0]

            # generate calibration mask
            # flag first and last integration of each mask, as well one before
            # and one after, to catch off->on and on->off transitions
            trans = np.zeros_like(time_series, dtype=bool)
            mask = np.zeros_like(time_series, dtype=bool)
            for i in idx:
                st = max(0, i - duration_window // 2)
                en = min(len(interp_times) - 1, i + duration_window // 2 + 1)
                # get closest integrations
                sti = np.argmin(np.abs(interp_times[st] - scantimes))
                eni = np.argmin(np.abs(interp_times[en] - scantimes)) + 1
                mask[sti:eni] = True
                # set flags
                trans[sti] = True
                trans[eni - 1] = True
                if sti > 0:
                    trans[sti - 1] = True
                if eni < len(mask):
                    trans[eni] = True

            # flag cal transitions
            if verbose:
                print("Saving cal and flag mask...")
            for i in range(data.shape[0]):
                if trans[i]:
                    flag[i, :] = True

            # save cal state
            metadata["CAL"] = mask
            sdhdf["data"]["beam_0"]["band_SB0"][scan]["metadata"][:] = metadata


def main():
    parser = argparse.ArgumentParser(
        description="Identify calibration integrations",
        prog="findcal.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    parser.add_argument(
        "-s", "--start", type=int, default=0, help="First channel of integration range",
    )
    parser.add_argument(
        "-e", "--end", type=int, default=-1, help="Last channel of integration range",
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=30.0, help="Cal-on duration (seconds)",
    )
    parser.add_argument(
        "-p", "--period", type=float, default=60.0, help="Cal period",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose information",
    )
    args = parser.parse_args()
    findcal(
        args.datafile,
        start=args.start,
        end=args.end,
        duration=args.duration,
        period=args.period,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
