"""
flagtime.py
Identify and flag interference in a SDHDF data file along
the time axis using multiprocessing.

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
Trey Wenger - March 2022
"""

import argparse
import h5py
import numpy as np
import time
import multiprocessing as mp
import warnings
import gc
import os

from . import __version__
from .utils import add_history
from .flagutils import generate_flag_mask

_MAX_INT = np.iinfo(int).max


def worker(inqueue, outqueue, datafile, scan, batchsize, chanbin, window, cutoff, grow):
    """
    Multiprocessing worker. Handles batch preparation and flag generation.
    """
    print(f"Launching worker with PID: {os.getpid()}")
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")

    # Chunk cache size = 1 GB ~ 67 default chunks
    # cache_size = 1024 ** 3 * 1
    cache_size = None
    with h5py.File(datafile, "r", rdcc_nbytes=cache_size) as sdhdf:
        # get data and mask
        data = sdhdf["data"]["beam_0"]["band_SB0"][scan]["data"]
        flag = sdhdf["data"]["beam_0"]["band_SB0"][scan]["flag"]
        metadata = sdhdf["data"]["beam_0"]["band_SB0"][scan]["metadata"]
        calon = metadata["CAL"]

        # storage for batch processing
        data_batch = np.zeros((data.shape[0], data.shape[1], batchsize), dtype=float)
        mask_batch = np.zeros((data.shape[0], batchsize), dtype=bool)

        # storage for averaging
        data_buffer = np.zeros((data.shape[0], data.shape[1]), dtype=float)
        isnan_buffer = np.zeros((chanbin, data.shape[0], data.shape[1]), dtype=bool)
        index_buffer = np.ones(chanbin, dtype=int) * -1

        while True:
            # get data from queue
            queue = inqueue.get(block=True)

            # catch kill worker
            if queue is None:
                print(f"Killing worker with PID: {os.getpid()}")
                break

            # Unpack index range from queue
            startidx, endidx = queue

            # loop over channels
            batch = 0
            for i in range(startidx, endidx):
                # get mask for this channel
                mask = flag[:, i]

                # get channel range
                start = max(0, i - chanbin // 2)
                end = min(data.shape[2], i + chanbin // 2 + 1)

                # remove old integrations from buffer
                for buffer_idx, idx in enumerate(index_buffer):
                    if idx > -1 and idx < start:
                        data_buffer = np.nansum([data_buffer, -data[:, :, idx]], axis=0)
                        isnan_buffer[buffer_idx] = np.zeros_like(
                            isnan_buffer[buffer_idx]
                        )
                        index_buffer[buffer_idx] = -1

                # add new integrations to buffer
                for idx in range(start, end):
                    if idx not in index_buffer:
                        first_empty = np.where(index_buffer == -1)[0][0]
                        data_buffer = np.nansum([data_buffer, data[:, :, idx]], axis=0)
                        isnan_buffer[first_empty] = np.isnan(data[:, :, idx])
                        index_buffer[first_empty] = idx

                # Calculate mean in buffer
                good = np.where(index_buffer != -1)[0]
                count_notnan = np.sum(~isnan_buffer[good], axis=0)
                count_notnan[count_notnan == 0] = _MAX_INT
                dat = data_buffer / count_notnan

                # apply mask
                dat[np.repeat(mask[:, None], 4, axis=1)] = np.nan

                # Estimate median cal-on and cal-off power, normalize to constant power
                median_calon = np.nanmedian(dat[calon], axis=0)
                median_caloff = np.nanmedian(dat[~calon], axis=0)
                dat[calon] = dat[calon] - (median_calon - median_caloff)

                # add to batch
                data_batch[:, :, batch] = dat
                mask_batch[:, batch] = mask
                batch += 1

            # submit batch for processing
            outmask = generate_flag_mask(
                data_batch[:, :, :batch],
                mask_batch[:, :batch],
                window=window,
                cutoff=cutoff,
                grow=grow,
            )

            # send result to output queue
            outqueue.put([startidx, endidx, outmask])


def flagtime(
    datafile,
    chanbin=1,
    batchsize=20000,
    window=11,
    cutoff=5.0,
    grow=0.75,
    num_cpus=None,
    verbose=False,
):
    """
    Flag an SDHDF dataset along the time axis.

    Inputs:
        datafile :: string
            SDHDF file
        chanbin :: odd integer
            Average over this many channels before flagging
        batchsize :: integer
            Process in batches of this many channels
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
    if chanbin % 2 == 0:
        raise ValueError("chanbin must be odd")
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
        add_history(sdhdf, f"JAG-PIPELINE-FLAGTIME VERSION: {__version__}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGTIME TIMEBIN: {chanbin}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGTIME BATCHSIZE: {batchsize}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGTIME WINDOW: {window}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGTIME CUTOFF: {cutoff}")
        add_history(sdhdf, f"JAG-PIPELINE-FLAGTIME GROW: {grow}")

        # get total number of channels
        num_chans = 0
        complete = 0
        scans = [
            key for key in sdhdf["data"]["beam_0"]["band_SB0"].keys() if "scan" in key
        ]
        scans = sorted(scans, key=lambda scan: int(scan[5:]))
        for scan in scans:
            num_chans += sdhdf["data"]["beam_0"]["band_SB0"][scan]["data"].shape[2]

        # Loop over scans
        starttime = time.time()
        for scan in scans:
            print()
            print(scan)
            print()
            data = sdhdf[f"data/beam_0/band_SB0/{scan}/data"]
            flag = sdhdf[f"data/beam_0/band_SB0/{scan}/flag"]
            metadata = sdhdf[f"data/beam_0/band_SB0/{scan}/metadata"]

            # Check that some cal-on data are present
            if np.all(~metadata["CAL"]):
                print(f"WARNING: {scan} has no cal-signal-on data")

            # skip one-integration scans
            if data.shape[0] == 1:
                print(f"WARNING: Skipping {scan} which has only one integration")
                complete += data.shape[2]
                continue

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
                    chanbin,
                    window,
                    cutoff,
                    grow,
                ),
            )

            # submit jobs to queue
            running = 0
            for start in range(0, data.shape[2], batchsize):
                end = min(data.shape[2], start + batchsize)
                inqueue.put([start, end])
                running += 1

            # wait for queue to empty
            while running > 0:
                start, end, mask = outqueue.get()
                flag[:, start:end] = mask
                complete += end - start
                running -= 1
                if verbose:
                    runtime = time.time() - starttime
                    timeper = runtime / (complete + 1)
                    remaining = timeper * (num_chans - complete)
                    print(
                        f"Flagging Channel: {complete}/{num_chans} "
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
            gc.collect()
            print()
            print("Collecting")
            print()

        # Done
        if verbose:
            runtime = time.time() - starttime
            print(
                f"Flagging Channel: {complete}/{num_chans} "
                + f"Runtime: {runtime:.2f} s                     "
            )


def main():
    parser = argparse.ArgumentParser(
        description="Automatically flag SDHDF along frequency axis",
        prog="flagtime.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    parser.add_argument(
        "--chanbin",
        type=int,
        default=1,
        help="Average this number of channels before flagging",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=20000,
        help="Process in batches of this many channels",
    )
    parser.add_argument(
        "--window", type=int, default=11, help="Rolling window size",
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
    flagtime(
        args.datafile,
        chanbin=args.chanbin,
        batchsize=args.batchsize,
        window=args.window,
        cutoff=args.cutoff,
        grow=args.grow,
        num_cpus=args.num_cpus,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
