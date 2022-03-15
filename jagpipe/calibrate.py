"""
calibrate.py
Calibrate a SDHDF dataset.

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
Trey Wenger - January 2022
"""

import argparse
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from . import __version__
from .utils import add_history
from .flagchan import generate_flag_mask


def linear(x, slope, intercept):
    return intercept + slope * x


def init_cal_datasets(scan, num_freq):
    """
    Add new datasets to the scan group to store:
        1. Calibrated data
        2. System temperature
        3. System gain
    These datasets are deleted if they already exist.

    Inputs:
        scan :: h5py.Group
            Scan group
        num_freq :: integer
            The number of frequency channels

    Returns: caldata, tsysdata, gaindata
        caldata, tsysdata, gaindata :: h5py.Dataset
            The newly created datasets.
    """
    # Chunk shape for data storage
    # For float, this is about 12 MB
    chunk_time = 20
    chunk_pol = 4
    chunk_freq = min(20000, num_freq)
    chunks = (chunk_time, chunk_pol, chunk_freq)

    # create calibrated data dataset
    if "calibrated" in scan.keys():
        del scan["calibrated"]
    dat = np.empty((0, 4, num_freq), dtype=float)
    caldata = scan.create_dataset(
        "calibrated", data=dat, maxshape=(None, 4, num_freq), chunks=chunks,
    )
    # copy attributes from data
    for attr in scan["data"].attrs.keys():
        caldata.attrs[attr] = scan["data"].attrs[attr]
    # Set attributes
    caldata.attrs["TYPE"] = "Antenna Temperature"
    caldata.attrs["UNIT"] = "K"

    # create Tsys dataset
    if "system_temperature" in scan.keys():
        del scan["system_temperature"]
    dat = np.empty((2, num_freq), dtype=float)
    tsysdata = scan.create_dataset(
        "system_temperature", data=dat, maxshape=(2, num_freq), chunks=(2, chunk_freq),
    )
    tsysdata.attrs["NAME"] = "system_temperature"
    tsysdata.attrs["DESCRIPTION"] = "Cal-off system temperature"
    tsysdata.attrs["DIMENSION_LABELS"] = ["polarization", "channel"]
    tsysdata.attrs["UNIT"] = "K"

    # create system gain dataset
    if "system_gain" in scan.keys():
        del scan["system_gain"]
    dat = np.empty((2, num_freq), dtype=float)
    gaindata = scan.create_dataset(
        "system_gain", data=dat, maxshape=(2, num_freq), chunks=(2, chunk_freq),
    )
    gaindata.attrs["NAME"] = "system_gain"
    gaindata.attrs["DESCRIPTION"] = "System gain"
    gaindata.attrs["DIMENSION_LABELS"] = ["polarization", "channel"]
    gaindata.attrs["UNIT"] = "Counts per K"
    return caldata, tsysdata, gaindata


def get_avg_cal_spectra(
    data,
    metadata,
    flag,
    scani,
    num_scans,
    flagwindow=101,
    flagcutoff=5.0,
    verbose=False,
):
    """
    Get CAL-ON and CAL-OFF spectra averaged over the scan. Automatically flag
    the spectra.

    Inputs:
        data :: h5py.Dataset
            Scan data
        metadata :: h5py.Dataset
            Scan metadata
        flag :: h5py.Dataset
            Scan flag table
        scani :: integer
            The scan number
        num_scans :: integer
            Number of scans in dataset
        flagwindow :: integer
            Rolling window size for automatic flagging
        flagcutoff :: scalar
            Sigma clip for automatic flagging
        verbose :: boolean
            If True, print information

    Returns: avg_calon, avg_caloff
        avg_calon, avg_caloff :: (4, N) arrays of scalars
            Average CAL-ON and CAL-OFF spectra
    """
    # Average CAL-ON and CAL-OFF spectra
    avg_calon = np.zeros(data.shape[1:], dtype=float)
    count_calon = np.zeros(data.shape[2], dtype=int)
    avg_caloff = np.zeros(data.shape[1:], dtype=float)
    count_caloff = np.zeros(data.shape[2], dtype=int)
    for i in range(data.shape[0]):
        if verbose and i % 10 == 0:
            print(
                f"Reading Scan {scani}/{num_scans}     "
                + f"Integration {i}/{data.shape[0]}   ",
                end="\r",
            )
        # apply flag
        flg = np.repeat(flag[i][None, :], 4, axis=0)
        dat = data[i, :]
        dat[flg] = np.nan

        if metadata[i]["CAL"]:
            avg_calon = np.nansum([avg_calon, dat], axis=0)
            count_calon += ~flag[i]
        else:
            avg_caloff = np.nansum([avg_caloff, dat], axis=0)
            count_caloff += ~flag[i]
    avg_calon /= count_calon
    avg_caloff /= count_caloff

    # flag calon
    mask = np.any(np.isnan(avg_calon), axis=0)
    mask = generate_flag_mask(
        avg_calon[None].T, mask[None].T, window=flagwindow, cutoff=flagcutoff
    ).T[0]
    avg_calon[np.repeat(mask[None, :], 4, axis=0)] = np.nan

    # flag caloff
    mask = np.any(np.isnan(avg_caloff), axis=0)
    mask = generate_flag_mask(
        avg_caloff[None].T, mask[None].T, window=flagwindow, cutoff=flagcutoff
    ).T[0]
    avg_caloff[np.repeat(mask[None, :], 4, axis=0)] = np.nan
    return avg_calon, avg_caloff


def calc_system_gain(
    avg_calon, avg_caloff, tcal_xx, tcal_yy, frequency, smoothwidth=1001
):
    """
    Calculate the average system gain for a scan using the average cal
    deflection and assumed calibration noise temperature. Smooth the
    system gain spectrum.

    Inputs:
        avg_calon, avg_caloff :: (4, N) arrays of scalars
            Scan-average CAL-ON and CAL-OFF spectra
        tcal_xx, tcal_yy :: (N,) arrays of scalars
            Assumed XX and YY calibration noise temperature (K)
        frequency :: (N,) array of scalars
            Channel frequencies (Hz)
        smoothwidth :: integer
            Kernel width for smoothing

    Returns: gain_xx, gain_yy
        gain_xx, gain_yy :: (N,) arrays of scalars
            System gain (Counts per K)
    """
    # system gain
    gain_xx = (avg_calon - avg_caloff)[0] / tcal_xx
    gain_yy = (avg_calon - avg_caloff)[1] / tcal_yy

    # sinc filter
    filt_chans = np.arange(6 * smoothwidth + 1) - 3 * smoothwidth
    filt = np.sinc(2.0 * filt_chans / smoothwidth)
    filt /= filt.sum()

    # Smooth XX gain, handling nans
    isnan = np.isnan(gain_xx)
    nandata = np.ones_like(gain_xx)
    gain_xx[isnan] = 0.0
    nandata[isnan] = 0.0
    gain_xx = np.convolve(gain_xx, filt, mode="same")
    nandata = np.convolve(nandata, filt, mode="same")
    gain_xx = gain_xx / nandata
    gain_xx[nandata < 0.5] = np.nan

    # Smooth YY gain, handling nans
    isnan = np.isnan(gain_yy)
    nandata = np.ones_like(gain_yy)
    gain_yy[isnan] = 0.0
    nandata[isnan] = 0.0
    gain_yy = np.convolve(gain_yy, filt, mode="same")
    nandata = np.convolve(nandata, filt, mode="same")
    gain_yy = gain_yy / nandata
    gain_yy[nandata < 0.5] = np.nan

    # Interpolate gains
    isnan = np.isnan(gain_xx)
    gain_xx = interp1d(frequency[~isnan], gain_xx[~isnan], fill_value="extrapolate",)(
        frequency
    )
    isnan = np.isnan(gain_yy)
    gain_yy = interp1d(frequency[~isnan], gain_yy[~isnan], fill_value="extrapolate",)(
        frequency
    )
    return gain_xx, gain_yy


def calc_phase_trajectory(avg_calon, avg_caloff, frequency):
    """
    Fit the XX, YY phase delay and return the phase trajectory
    (i.e., phase vs. channel).

    Inputs:
        avg_calon, avg_caloff :: (4, N) arrays of scalars
            Scan-average CAL-ON and CAL-OFF spectra
        frequency :: (N,) array of scalars
            Channel frequencies (Hz)

    Returns:
        phase_fit :: (N,) array of scalars
            Fitted XX, YY phase at each channel
    """
    # calculate phase arctan(YX/XY)
    theta = np.arctan((avg_calon - avg_caloff)[3] / (avg_calon - avg_caloff)[2])
    theta[~np.isnan(theta)] = np.unwrap(theta[~np.isnan(theta)], period=np.pi)

    # fit phase delay using robust least squares
    isnan = np.isnan(theta)
    popt, pcov = curve_fit(
        linear,
        frequency[~isnan] / 1.0e6,
        theta[~isnan],
        p0=[0.0, 0.0],
        method="trf",
        loss="soft_l1",
    )
    phase_fit = linear(frequency / 1.0e6, *popt)
    return phase_fit


def calibrate(
    datafile, tcalfile, flagwindow=101, flagcutoff=5.0, smoothwidth=1001, verbose=False,
):
    """
    Calibrate a SDHDF dataset.
        1. Use calibration noise deflection to determine system gain.
        2. Use assumed calibration noise temperature to determine antenna temperature.
        3. Determine system temperature.
        4. Remove cross-polarization phase delay.
    Save results to new tables: caldata, system_temperature, system_gain

    Inputs:
        datafile :: string
            Uncalibrated SDHDF file
        tcalfile :: string
            File containing calibration noise temperature data
        flagwindow :: integer
            Rolling window size for automatic flagging
        flagcutoff :: scalar
            Sigma clip for automatic flagging
        smoothwidth :: integer
            Kernel width for smoothing
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    if flagwindow % 2 == 0:
        raise ValueError("window must be odd")

    # Load Tcal data
    tcal_data = np.genfromtxt(tcalfile, dtype=None, encoding="utf-8")
    calc_tcal_xx = interp1d(tcal_data[:, 0], tcal_data[:, 1], fill_value="extrapolate")
    calc_tcal_yy = interp1d(tcal_data[:, 0], tcal_data[:, 2], fill_value="extrapolate")

    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        # add history items
        add_history(sdhdf, f"JAG-PIPELINE-CALIBRATE VERSION: {__version__}")
        add_history(sdhdf, f"JAG-PIPELINE-CALIBRATE TCALFILE: {tcalfile}")
        add_history(sdhdf, f"JAG-PIPELINE-CALIBRATE FLAGWINDOW: {flagwindow}")
        add_history(sdhdf, f"JAG-PIPELINE-CALIBRATE FLAGCUTOFF: {flagcutoff}")
        add_history(sdhdf, f"JAG-PIPELINE-CALIBRATE SMOOTHWIDTH: {smoothwidth}")

        # Loop over beams
        for beam in sdhdf["data"].keys():
            # Loop over bands
            for band in sdhdf["data"][beam].keys():
                # get frequency
                frequency = sdhdf["data"][beam][band]["frequency"][:]

                # get tcal
                tcal_xx = calc_tcal_xx(frequency / 1.0e6)
                tcal_yy = calc_tcal_yy(frequency / 1.0e6)
                mean_tcal = np.sqrt(tcal_xx * tcal_yy)

                # loop over scans
                scans = [
                    key for key in sdhdf["data"][beam][band].keys() if "scan_" in key
                ]
                scans = sorted(scans, key=lambda scan_id: int(scan_id[5:]))
                for scani, scan in enumerate(scans):
                    # initialize new datasets
                    caldata, tsysdata, gaindata = init_cal_datasets(
                        sdhdf["data"][beam][band][scan], len(frequency)
                    )

                    # get data, flags, and metadata
                    data = sdhdf["data"][beam][band][scan]["data"]
                    flag = sdhdf["data"][beam][band][scan]["flag"]
                    metadata = sdhdf["data"][beam][band][scan]["metadata"]

                    # Check that some cal-on data are present
                    if np.all(~metadata["CAL"]):
                        print(f"WARNING: {scan} has no cal-signal-on data")

                    # skip one-integration scans
                    if data.shape[0] == 1:
                        print(
                            f"WARNING: Skipping {scan} which has only one integration"
                        )
                        continue

                    # Get average CAL-ON and CAL-OFF sepctra
                    avg_calon, avg_caloff = get_avg_cal_spectra(
                        data,
                        metadata,
                        flag,
                        scani,
                        len(scans),
                        flagwindow=flagwindow,
                        flagcutoff=flagcutoff,
                        verbose=verbose,
                    )

                    # Calculate system gain
                    gain_xx, gain_yy = calc_system_gain(
                        avg_calon,
                        avg_caloff,
                        tcal_xx,
                        tcal_yy,
                        frequency,
                        smoothwidth=smoothwidth,
                    )

                    # system gain geometric mean
                    gain_mean = np.sqrt(gain_xx * gain_yy)

                    # cal-off system temperature
                    tsys_xx = avg_caloff[0] / gain_xx
                    tsys_yy = avg_caloff[1] / gain_yy

                    # Fit XX, YY phase delay and get XX, YY phase trajectory
                    phase_fit = calc_phase_trajectory(avg_calon, avg_caloff, frequency)

                    # save tsys
                    tsysdata[:] = np.vstack([tsys_xx, tsys_yy])

                    # save gain
                    gaindata[:] = np.vstack([gain_xx, gain_yy])

                    # save calibrated data
                    for i in range(data.shape[0]):
                        if verbose and i % 10 == 0:
                            print(
                                f"Calibrating Scan {scani}/{len(scans)}     "
                                + f"Integration {i}/{data.shape[0]}   ",
                                end="\r",
                            )
                        xx, yy, xy, yx = data[i, :]
                        # XX YY gain
                        xx /= gain_xx
                        yy /= gain_yy
                        # XY YX delay
                        cross = xy + 1.0j * yx
                        cross *= np.exp(-1.0j * phase_fit)
                        xy = np.real(cross) / gain_mean
                        yx = np.imag(cross) / gain_mean
                        # remove cal signal if necessary
                        if metadata[i]["CAL"]:
                            xx -= tcal_xx
                            yy -= tcal_yy
                            xy -= mean_tcal
                            yx -= mean_tcal
                        # save
                        caldata.resize(caldata.shape[0] + 1, axis=0)
                        caldata[-1] = np.vstack([xx, yy, xy, yx])


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate an SDHDF dataset",
        prog="calibrate.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    parser.add_argument(
        "tcalfile", type=str, help="Calibration noise temperature data file",
    )
    parser.add_argument(
        "--flagwindow",
        type=int,
        default=101,
        help="Rolling window size for automatic flagging",
    )
    parser.add_argument(
        "--flagcutoff",
        type=float,
        default=5.0,
        help="Sigma threshold for automatic flagging",
    )
    parser.add_argument(
        "--smoothwidth", type=int, default=1001, help="Smoothing kernel width",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose information",
    )
    args = parser.parse_args()
    calibrate(
        args.datafile,
        args.tcalfile,
        flagwindow=args.flagwindow,
        flagcutoff=args.flagcutoff,
        smoothwidth=args.smoothwidth,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
