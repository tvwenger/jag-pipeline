"""
flagchan.py
Identify and flag interference in a SDHDF data file along
the frequency axis using multiprocessing.

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
import queue
import multiprocessing as mp
import time

from . import __version__
from .utils import add_history

_MAX_INT = np.iinfo(int).max


def gradient_filter(data, delta):
    """
    Use a first and second order finite differences filter to
    identify narrow interference spikes.
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
    # second order
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
    newdata = np.concatenate([data[:halfwin][::-1], data, data[-halfwin:][::-1]])
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
    # Catch all nan
    if np.all(np.isnan(data)):
        mask[:] = True
        return mask

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


def worker(inqueue, outqueue, window, cutoff):
    # Multiprocessing worker
    while True:
        # get data from queue
        data = inqueue.get(block=True)
        if data is None:
            break
        # Unpack
        idx, dat, mask = data
        # Process
        flag = generate_flag_mask(dat, mask, window=window, cutoff=cutoff)
        # Save
        outqueue.put([idx, flag])


def flagchan(
    datafile, timebin=1, window=101, cutoff=5.0, num_cpus=None, verbose=False,
):
    """
    Read data file, apply flagging, save flag data.

    Inputs:
        datafile :: string
            SDHDF file
        timebin :: odd integer
            Average over this many integrations before flagging
        window :: integer
            Rolling window size along frequency axis
        cutoff :: scalar
            Sigma clipping threshold
        num_cpus :: integer
            Number of CPUs to use in multiprocesing. If None, use all.
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    if timebin % 2 == 0:
        raise ValueError("timebins must be odd")
    if window % 2 == 0:
        raise ValueError("window must be odd")

    # Default CPU count
    if num_cpus is None:
        num_cpus = mp.cpu_count()

    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        # add history items
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN VERSION: {__version__}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN TIMEBIN: {timebin}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN WINDOW: {window}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN CUTOFF: {cutoff}")

        # get total number of integrations
        num_int = 0
        complete = 0
        scans = [
            key for key in sdhdf["data"]["beam_0"]["band_SB0"].keys() if "scan" in key
        ]
        scans = sorted(scans, key=lambda scan: int(scan[6:]))
        for scan in scans:
            num_int += sdhdf["data"]["beam_0"]["band_SB0"][scan]["data"].shape[0]

        # initialize queues
        inqueue = mp.Queue()
        outqueue = mp.Queue()

        # initialize the pool
        pool = mp.Pool(num_cpus, worker, (inqueue, outqueue, window, cutoff))

        # Loop over scans
        starttime = time.time()
        for scan in scans:
            # get data and mask
            data = sdhdf["data"]["beam_0"]["band_SB0"][scan]["data"]
            flag = sdhdf["data"]["beam_0"]["band_SB0"][scan]["flag"]

            # storage for data, flag
            data_buffer = np.ones((data.shape[1], data.shape[2]), dtype=float)
            isnan_buffer = np.zeros((timebin, data.shape[1], data.shape[2]), dtype=bool)
            index_buffer = np.ones(timebin, dtype=int) * -1

            # loop over integrations
            running = 0
            for i in range(data.shape[0]):
                if verbose and complete > 0 and (complete % 10 == 0):
                    runtime = time.time() - starttime
                    timeper = runtime / (complete + 1)
                    remaining = timeper * (num_int - complete)
                    print(
                        f"Flagging integration: {complete}/{num_int} "
                        + f"ETA: {remaining:0.1f} s          ",
                        end="\r",
                    )
                # get mask, skip if all data are flagged
                mask = flag[i, :]
                if np.all(mask):
                    continue

                # get integration range
                start = max(0, i - timebin // 2)
                end = min(data.shape[0], i + timebin // 2 + 1)

                # remove old integrations from buffer
                bad = np.where(index_buffer < start)[0]
                index_buffer[bad] = -1

                # add new integrations to buffer
                for idx in range(start, end):
                    if idx not in index_buffer:
                        first_empty = np.where(index_buffer == -1)[0][0]
                        data_buffer = np.nansum([data_buffer, data[idx, :, :]], axis=0)
                        isnan_buffer[first_empty] = np.isnan(data[idx, :, :])
                        index_buffer[first_empty] = idx

                # Calculate mean in buffer
                good = np.where(index_buffer != -1)[0]
                count_notnan = np.sum(~isnan_buffer[good], axis=0)
                count_notnan[count_notnan == 0] = _MAX_INT
                dat = data_buffer / count_notnan

                # apply mask
                dat[np.repeat(mask[None, :], 4, axis=0)] = np.nan

                # add to queue
                inqueue.put([i, dat, mask])
                running += 1

                # wait for queue to empty
                while running >= num_cpus:
                    try:
                        idx, flg = outqueue.get_nowait()
                        flag[idx, :] = flg
                        complete += 1
                        running -= 1
                    except queue.Empty:
                        pass

            # wait for remaining processes to finish
            while running > 0:
                idx, flg = outqueue.get()
                flag[idx, :] = flg
                complete += 1
                running -= 1
        if verbose:
            runtime = time.time() - starttime
            print(
                f"Flagging integration: {complete}/{num_int} "
                + f"Runtime: {runtime:.2f} s                     "
            )

        # terminate processess
        for _ in range(num_cpus):
            inqueue.put(None)

        # close queues and pool
        inqueue.close()
        inqueue.join_thread()
        outqueue.close()
        outqueue.join_thread()
        pool.close()
        pool.join()


def main():
    parser = argparse.ArgumentParser(
        description="Automatically flag SDHDF along frequency axis",
        prog="flagchan.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    parser.add_argument(
        "-t",
        "--timebin",
        type=int,
        default=1,
        help="Average this number of integrations before flagging",
    )
    parser.add_argument(
        "-w", "--window", type=int, default=101, help="Rolling channel window size",
    )
    parser.add_argument(
        "-c", "--cutoff", type=float, default=5.0, help="Sigma clipping threshold",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        help="Number of CPUs to use in multiprocessing",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose information",
    )
    args = parser.parse_args()
    flagchan(
        args.datafile,
        timebin=args.timebin,
        window=args.window,
        cutoff=args.cutoff,
        num_cpus=args.num_cpus,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
