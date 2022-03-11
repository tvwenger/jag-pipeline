"""
waterfall.py
Generate waterfall figures from SDHDF data.

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
import warnings
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from . import __version__

# Force non-GUI backend
mpl.use("Agg")


def waterfall(
    datafile, prefix, scans=None, chanbin=None, showcal=False, plotflagged=False,
):
    """
    Generate waterfall figure from a given SDHDF datafile.

    Inputs:
        datafile :: string
            SDHDF filename
        prefix :: string
            Filenames saved like "{prefix}_scan{num}_pseudoI.png"
        scans :: list of integers
            Scans to plot
        chanbin :: integer
            Number of channels to bin. If None, set so that there are
            ~4,000 channel bins across the display range.
        showcal :: boolean
            If True, highlight cal-on integrations
        plotflagged :: boolean
            If True, plot flagged data

    Returns: Nothing
    """
    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r", rdcc_nbytes=cache_size) as sdhdf:
        # get scans
        if scans is None:
            scans = [
                key
                for key in sdhdf["data"]["beam_0"]["band_SB0"].keys()
                if "scan" in key
            ]
        else:
            scans = [
                f"scan_{scan}"
                for scan in scans
                if f"scan_{scan}" in sdhdf["data"]["beam_0"]["band_SB0"].keys()
            ]
        scans = sorted(scans, key=lambda scan: int(scan[5:]))

        # Get size of frequency and integration axes
        frequency = sdhdf["data"]["beam_0"]["band_SB0"]["frequency"][()]
        freqframe = sdhdf["data"]["beam_0"]["band_SB0"]["frequency"].attrs["FRAME"]

        # Set default values for bins if necessary
        if chanbin is None:
            chanbin = max(1, int(len(frequency) / 4000))
            print(f"chanbin: {chanbin}")

        # Correlations
        corr_idxs = [0, 1, 2, 3]
        datatypes = ["XX", "YY", "Re(XY)", "Im(XY)"]
        labels = ["XX", "YY", "ReXY", "ImXY"]

        for scani, scan in enumerate(scans):
            # get data
            exposure = sdhdf["data"]["beam_0"]["band_SB0"].attrs["EXPOSURE"]
            data = sdhdf["data"]["beam_0"]["band_SB0"][scan]["data"]
            flag = sdhdf["data"]["beam_0"]["band_SB0"][scan]["flag"]
            metadata = sdhdf["data"]["beam_0"]["band_SB0"][scan]["metadata"]
            scantimes = metadata["MJD"] * 24.0 * 3600.0
            num_int = data.shape[0]

            # plot extent
            start_mjd = scantimes[0]
            start_time = scantimes[0] - start_mjd
            end_time = scantimes[-1] - start_mjd + exposure
            chanwidth = frequency[1] - frequency[0]
            start_freq = (frequency[0] - chanwidth / 2.0) / 1e6
            end_freq = (frequency[-1] + chanwidth / 2.0) / 1e6
            scan_duration = end_time - start_time
            extent = [start_freq, end_freq, end_time, start_time]

            # Storage for plot data
            plot_num_freq = int(np.ceil(len(frequency) / chanbin))
            plot_num_int = int(np.round(scan_duration / exposure))
            plottimes = np.arange(plot_num_int) * exposure
            plotdata = np.ones((4, plot_num_int, plot_num_freq)) * np.nan

            for i in range(num_int):
                if i % 10 == 0:
                    print(
                        f"Scan {scan} ({scani}/{len(scans)-1})     "
                        + f"Integration {i}/{num_int}   ",
                        end="\r",
                    )

                # get closest plot index for this integration
                ploti = np.argmin(np.abs(scantimes[i] - start_mjd - plottimes))

                # Loop over correlations
                for corr_idx in corr_idxs:
                    # bin in frequency
                    dat = data[i, corr_idx, :]
                    if not plotflagged:
                        dat[flag[i]] = np.nan
                    pad = len(dat) % chanbin
                    if pad != 0:
                        pad = chanbin - pad
                    dat = np.pad(dat, (0, pad), mode="constant", constant_values=np.nan)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action="ignore", message="Mean of empty slice"
                        )
                        dat = np.nanmean(dat.reshape(-1, chanbin), axis=1)
                    plotdata[corr_idx, ploti] = dat

            # Generate figures
            for corr_idx, datatype, label in zip(corr_idxs, datatypes, labels):
                fig, ax = plt.subplots(figsize=(12, 12),)
                vmin, vmax = np.nanpercentile(plotdata[corr_idx], q=[5.0, 95.0])
                cax = ax.imshow(
                    plotdata[corr_idx],
                    extent=extent,
                    interpolation="none",
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(cax, ax=ax, label="Relative Power")

                # highligh cal-on integrations
                if showcal:
                    freq_width = 0.01 * (end_freq - start_freq)
                    first_int = None
                    last_int = None
                    for row, scantime in zip(metadata, scantimes):
                        if first_int is None and row["CAL"]:
                            first_int = scantime - start_mjd
                        elif (
                            first_int is not None
                            and last_int is None
                            and (not row["CAL"] or row == metadata[-1])
                        ):
                            last_int = scantime - start_mjd
                            # plot
                            rect = Rectangle(
                                (start_freq, first_int),
                                freq_width,
                                (last_int - first_int),
                                linewidth=0,
                                fill=True,
                                color="red",
                            )
                            ax.add_patch(rect)
                            rect = Rectangle(
                                (end_freq - freq_width, first_int),
                                freq_width,
                                (last_int - first_int),
                                linewidth=0,
                                fill=True,
                                color="red",
                            )
                            ax.add_patch(rect)
                            first_int = None
                            last_int = None
                    red_patch = Patch(color="red", label="CAL ON")
                    ax.legend(loc="upper left", handles=[red_patch])

                # labels and save
                ax.set_xlabel(f"{freqframe} Frequency (MHz)")
                ax.set_ylabel(f"Seconds Since MJD = {start_mjd:.6f}")
                ax.set_title(
                    "{0} {1} {2}".format(
                        datafile.replace("_", r"\_"),
                        scan.replace("_", r"\_"),
                        datatype,
                    )
                )
                fig.tight_layout()
                fname = f"{prefix}_{scan}_{label}.png"
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate waterfall figures from SDHDF file",
        prog="waterfall.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    parser.add_argument(
        "prefix", type=str, help="Output plot files prefix",
    )
    parser.add_argument(
        "-s",
        "--scans",
        type=int,
        nargs="+",
        default=None,
        help="Scans to plot. Default: all scans",
    )
    parser.add_argument(
        "-c",
        "--chanbin",
        type=int,
        default=None,
        help="Channel bin size. Default: 5000 bins across image",
    )
    parser.add_argument(
        "--showcal",
        action="store_true",
        default=False,
        help="Highlight cal-on integrations",
    )
    parser.add_argument(
        "--plotflagged", action="store_true", default=False, help="Plot flagged data",
    )
    args = parser.parse_args()
    waterfall(
        args.datafile,
        args.prefix,
        scans=args.scans,
        chanbin=args.chanbin,
        showcal=args.showcal,
        plotflagged=args.plotflagged,
    )


if __name__ == "__main__":
    main()
