"""
flagchan.py
Identify and flag interference in a SDHDF data file along
the frequency axis using multiprocessing.

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

import argparse
import h5py
import numpy as np
import time
import multiprocessing as mp
import warnings

from . import __version__
from .utils import add_history
from .flagutils import generate_flag_mask

_MAX_INT = np.iinfo(int).max


def worker(inqueue, outqueue, datafile, scan, batchsize, timebin, window, cutoff, grow):
    """
    Multiprocessing worker. Handles batch preparation and flag generation.
    """
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")

    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r", rdcc_nbytes=cache_size) as sdhdf:
        # get data and mask
        data = sdhdf["data"]["beam_0"]["band_SB0"][scan]["data"]
        flag = sdhdf["data"]["beam_0"]["band_SB0"][scan]["flag"]

        # storage for batch processing
        data_batch = np.zeros((batchsize, data.shape[1], data.shape[2]), dtype=float)
        mask_batch = np.zeros((batchsize, data.shape[2]), dtype=bool)

        # storage for averaging
        data_buffer = np.zeros((data.shape[1], data.shape[2]), dtype=float)
        isnan_buffer = np.zeros((timebin, data.shape[1], data.shape[2]), dtype=bool)
        index_buffer = np.ones(timebin, dtype=int) * -1

        while True:
            # get data from queue
            queue = inqueue.get(block=True)

            # catch kill worker
            if queue is None:
                break

            # Unpack index range from queue
            startidx, endidx = queue

            # loop over integrations
            batch = 0
            for i in range(startidx, endidx):
                # get mask for this integration
                mask = flag[i, :]

                # get integration range
                start = max(0, i - timebin // 2)
                end = min(data.shape[0], i + timebin // 2 + 1)

                # remove old integrations from buffer
                for buffer_idx, idx in enumerate(index_buffer):
                    if idx > -1 and idx < start:
                        data_buffer = np.nansum([data_buffer, -data[idx, :, :]], axis=0)
                        isnan_buffer[buffer_idx] = np.zeros_like(
                            isnan_buffer[buffer_idx]
                        )
                        index_buffer[buffer_idx] = -1

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

                # add to batch
                data_batch[batch] = dat
                mask_batch[batch] = mask
                batch += 1

            # submit batch for processing
            outmask = generate_flag_mask(
                data_batch[:batch].T,
                mask_batch[:batch].T,
                window=window,
                cutoff=cutoff,
                grow=grow,
            ).T

            # send result to output queue
            outqueue.put([startidx, endidx, outmask])


def flagchan(
    datafile,
    timebin=1,
    batchsize=20,
    window=21,
    cutoff=5.0,
    grow=0.75,
    num_cpus=None,
    verbose=False,
):
    """
    Flag an SDHDF dataset along the frequency axis.

    Inputs:
        datafile :: string
            SDHDF file
        timebin :: odd integer
            Average over this many integrations before flagging
        batchsize :: integer
            Process in batches of this many integrations
        window :: integer
            Rolling window size
        cutoff :: scalar
            Sigma clipping threshold
        grow :: scalar between 0.0 and 1.0
            Extend mask where more than grow*window adjacent data
            are masked
        num_cpus :: integer
            Number of CPUs to use in multiprocesing. If None, use all.
        verbose :: boolean
            If True, print information

    Returns: Nothing
    """
    if timebin % 2 == 0:
        raise ValueError("timebin must be odd")
    if window % 2 == 0:
        raise ValueError("window must be odd")

    # Default CPU count
    if num_cpus is None:
        num_cpus = mp.cpu_count()
    if num_cpus > mp.cpu_count():
        print(
            f"Requested num_cpus ({num_cpus}) exceeds avaiable cores {mp.cpu_count()}"
        )

    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r+", rdcc_nbytes=cache_size) as sdhdf:
        # add history items
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN VERSION: {__version__}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN TIMEBIN: {timebin}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN BATCHSIZE: {batchsize}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN WINDOW: {window}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN CUTOFF: {cutoff}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGCHAN GROW: {grow}")

        # get total number of integrations
        num_ints = 0
        complete = 0
        scans = [
            key for key in sdhdf["data"]["beam_0"]["band_SB0"].keys() if "scan" in key
        ]
        scans = sorted(scans, key=lambda scan: int(scan[5:]))
        for scan in scans:
            num_ints += sdhdf["data"]["beam_0"]["band_SB0"][scan]["data"].shape[0]

        # Loop over scans
        starttime = time.time()
        for scan in scans:
            data = sdhdf[f"data/beam_0/band_SB0/{scan}/data"]
            flag = sdhdf[f"data/beam_0/band_SB0/{scan}/flag"]

            # initialize queues
            inqueue = mp.Queue()
            outqueue = mp.Queue()

            # initialize the pool
            pool = mp.Pool(
                num_cpus,
                worker,
                (
                    inqueue,
                    outqueue,
                    datafile,
                    scan,
                    batchsize,
                    timebin,
                    window,
                    cutoff,
                    grow,
                ),
            )

            # submit jobs to queue
            running = 0
            for start in range(0, data.shape[0], batchsize):
                end = min(data.shape[0], start + batchsize)
                inqueue.put([start, end])
                running += 1

            # wait for queue to empty
            while running > 0:
                start, end, mask = outqueue.get()
                flag[start:end, :] = mask
                complete += end - start
                running -= 1
                if verbose:
                    runtime = time.time() - starttime
                    timeper = runtime / (complete + 1)
                    remaining = timeper * (num_ints - complete)
                    print(
                        f"Flagging Integration: {complete}/{num_ints} "
                        + f"ETA: {remaining:0.1f} s          ",
                        end="\r",
                    )

            # reset queues and pool
            for _ in range(num_cpus):
                inqueue.put(None)
            inqueue.close()
            inqueue.join_thread()
            outqueue.close()
            outqueue.join_thread()
            pool.close()
            pool.join()

        # Done
        if verbose:
            runtime = time.time() - starttime
            print(
                f"Flagging integration: {complete}/{num_ints} "
                + f"Runtime: {runtime:.2f} s                     "
            )


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
        "--timebin",
        type=int,
        default=1,
        help="Average this number of integrations before flagging",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=20,
        help="Process in batches of this many integrations",
    )
    parser.add_argument(
        "--window", type=int, default=21, help="Rolling window size",
    )
    parser.add_argument(
        "--cutoff", type=float, default=5.0, help="Sigma clipping threshold",
    )
    parser.add_argument(
        "--grow",
        type=float,
        default=0.75,
        help="Extend mask where more than grow*window adjacent data are masked",
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
        batchsize=args.batchsize,
        window=args.window,
        cutoff=args.cutoff,
        grow=args.grow,
        num_cpus=args.num_cpus,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
