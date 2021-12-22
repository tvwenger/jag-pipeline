"""
flagchan.py
Identify and flag interference in a SDHDF data file along
the frequency axis.

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
from astropy.time import Time
import h5py
import numpy as np
import time

from . import __version__

_MAX_INT = np.iinfo(int).max


def gradient_filter(data, delta):
    """
    Use a first order finite differences filter to identify
    narrow interference spikes.
    N.B. Data must be positive definite

    Inputs:
        data :: 1-D array of scalars
            Data

    Returns: mask
        mask :: 1-D array of boolean
            Mask (True = bad)
    """
    # First order
    foo = np.roll(data, -1 * delta)
    bar = np.roll(data, +1 * delta)
    grad = (bar - foo) / data
    med_grad = np.nanmedian(grad)
    rms_grad = 1.4826 * np.nanmedian(np.abs(grad - med_grad))
    mask = np.abs(grad - med_grad) > 5.0 * rms_grad
    grad = (bar - 2.0 * data + foo) / data
    med_grad = np.nanmedian(grad)
    rms_grad = 1.4826 * np.nanmedian(np.abs(grad - med_grad))
    mask = mask | (np.abs(grad - med_grad) > 5.0 * rms_grad)
    return mask


def rolling_sum(data, window):
    """
    Calculate rolling sum. Edges are handled by reflection.

    Inputs:
        data :: 1-D array of scalars
            Data
        window :: odd integer
            Window width

    Returns:
        std :: 1-D array of scalars
            Rolling standard deviation
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    halfwin = (window - 1) // 2
    newdata = np.concatenate(
        [data[:halfwin][::-1], data, data[-halfwin:][::-1]]
    )
    nansum = np.cumsum(~np.isnan(newdata))
    cumsum = np.nancumsum(newdata)
    foo = cumsum[window - 1 :] - np.concatenate([[0], cumsum[:-window]])
    bar = nansum[window - 1 :] - np.concatenate([[0], nansum[:-window]])
    bar[bar < 2] = _MAX_INT  # all nan
    return foo, bar


def rolling_mean(data, window):
    """
    Calculate rolling mean. Edges are handled by reflection.

    Inputs:
        data :: 1-D array of scalars
            Data
        window :: odd integer
            Window width

    Returns:
        mean :: 1-D array of scalars
            Rolling mean
    """
    foo, bar = rolling_sum(data, window)
    return foo / bar


def rolling_std(data, window):
    """
    Calculate rolling standard deviation. Edges are handled by reflection.
    N.B. This is an approximation with ~0.1% accuracy.

    Inputs:
        data :: 1-D array of scalars
            Data
        window :: odd integer
            Window width

    Returns:
        std :: 1-D array of scalars
            Rolling standard deviation
    """
    mean = rolling_mean(data, window)
    foo, bar = rolling_sum((data - mean) ** 2.0, window)
    return np.sqrt(foo / (bar - 1))


def generate_flag_mask(data, mask, window=101, cutoff=5.0):
    """
    Automatically flag a spectrum using
    1. gradient mask (narrow interference)
    2. sigma clipping (narrow inteference)
    3. rms clipping (broad interference)
    4. mask expansion (broad, low-level interference)

    Inputs:
        data :: 2-D array of scalars (shape: 4 x N)
            N-length data, with [XX, YY, XY or Re(XY), YX or Im(YX)]
            along the first axis.
        mask :: 1-D array of boolean (shape: N)
            Initial N-length boolean mask
        window :: integer
            Rolling window size (after channel binning)
        cutoff :: scalar
            Sigma clip

    Returns:
        new_mask :: 1-D array of boolean (shape: N)
            Updated N-length boolean mask
    """
    # flag narrow spikes in total power using gradient filter
    total_power = data[0] + data[1]
    gradient_mask = gradient_filter(total_power, 3)
    mask = mask | gradient_mask

    # rolling window mean on total power
    total_power[mask] = np.nan
    mean_tp = rolling_mean(total_power, window)

    # rolling window mean and rms on cross correlations.
    cross_power = 2.0 * data[2] + 2.0 * data[3]
    cross_power[mask] = np.nan
    mean_cross = rolling_mean(cross_power, window)
    rms_cross = rolling_std(cross_power, window)
    rms_cross[rms_cross == 0.0] = np.nan

    # flag narrow spikes in cross power
    norm_data = (cross_power - mean_cross) / rms_cross
    mask = mask | (np.abs(norm_data) > cutoff)

    # flag narrow spikes in total power
    norm_data = (total_power - mean_tp) / rms_cross
    mask = mask | (np.abs(norm_data) > cutoff)

    # flag broad noise based on rms
    norm_rms = rms_cross / mean_tp
    med = np.nanmedian(norm_rms)
    mad = np.nanmedian(np.abs(norm_rms - med))
    mask = mask | (np.abs(norm_rms - med) > cutoff * 1.4826 * mad)

    # flag channels where >25% of adjacent channels are flagged
    flagged = rolling_mean(mask, window)
    mask = mask | (flagged > 0.25)
    return mask


def flagchan(
    datafile,
    window=101,
    cutoff=5.0,
    verbose=False,
):
    """
    Read data file, apply flagging, save flag data.

    Inputs:
        datafile :: string
            SDHDF file
        window :: integer
            Rolling window size along frequency axis
        cutoff :: scalar
            Sigma clipping threshold
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        # initialize
        sdhdf["metadata"].attrs["JAG-PIPELINE-FLAGCHAN-VERSION"] = __version__
        sdhdf["metadata"].attrs[
            "JAG-PIPELINE-FLAGCHAN-EXECTIME"
        ] = Time.now().isot
        sdhdf["metadata"].attrs["JAG-PIPELINE-FLAGCHAN-WINDOW"] = window
        sdhdf["metadata"].attrs["JAG-PIPELINE-FLAGCHAN-CUTOFF"] = cutoff

        # get data and mask
        data = sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["data"]
        flag = sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["flag"]

        # Loop over times
        start = time.time()
        for i in range(data.shape[0]):
            if verbose:
                if i % 10 == 0:
                    runtime = time.time() - start
                    timeper = runtime / (i + 1)
                    remaining = timeper * (data.shape[0] - i)
                    print(
                        f"Flagging time: {i}/{data.shape[0]} "
                        + f"ETA: {remaining:0.1f} s          ",
                        end="\r",
                    )
            # apply mask, update flag
            dat = data[i, :, :]
            mask = flag[i, :]
            dat[np.repeat(mask[None, :], 4, axis=0)] = np.nan
            flag[i, :] = generate_flag_mask(
                dat, mask, window=window, cutoff=cutoff
            )
        if verbose:
            runtime = time.time() - start
            print(
                f"Flagging time: {data.shape[0]}/{data.shape[0]} "
                + f"Runtime: {runtime:.2f} s                     "
            )


def main():
    parser = argparse.ArgumentParser(
        description="Automatically flag SDHDF along frequency axis",
        prog="flagchan.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datafile",
        type=str,
        help="SDHDF file",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=101,
        help="Rolling channel window size",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose information",
    )
    args = parser.parse_args()
    flagchan(
        args.datafile,
        window=args.window,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
