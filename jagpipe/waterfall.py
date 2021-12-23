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
import h5py
import numpy as np
import matplotlib.pyplot as plt


def waterfall(
    datafile, prefix, skiptime=1, skipchan=1, plotcal=False, plotflagged=False,
):
    """
    Generate waterfall figure from a given SDHDF datafile, assuming data are
    contiguous in time!

    Inputs:
        datafile :: string
            SDHDF filename
        prefix :: string
            Filenames saved like "{prefix}_pseudoI.png"
        skiptime :: integer
            Downsample integrations by this amount (i.e., plot every
            skiptime integration)
        skipchan :: integer
            Downsample channels by this amount (i.e., plot every
            skipchan channel)
        plotcal :: boolean
            If True, plot cal-on integrations
        plotflagged :: boolean
            If True, plot flagged data

    Returns: Nothing
    """
    # Chunk cache size = 8 GB ~ 670 default chunks
    cache_size = 1024 ** 3 * 8
    with h5py.File(datafile, "r", rdcc_nbytes=cache_size) as sdhdf:
        position = sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["position"][::skiptime]
        frequency = sdhdf["data"]["beam_0"]["band_SB0"]["frequency"][::skipchan]
        freqtype = sdhdf["data"]["beam_0"]["band_SB0"]["frequency"].attrs["FRAME"]
        flag = sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["flag"]
        cal = sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["cal"][::skiptime]
        data = sdhdf["data"]["beam_0"]["band_SB0"]["scan_0"]["data"]

        extent = [
            frequency[0] / 1.0e6,
            frequency[-1] / 1.0e6,
            (position["MJD"][-1] - position["MJD"][0]) * (24 * 3600),
            0,
        ]
        datatypes = ["XX", "YY", "Re(XY)", "Im(XY)"]
        labels = ["XX", "YY", "ReXY", "ImXY"]

        for datatype, label in zip(datatypes, labels):
            # get data
            print(f"Reading {datatype}")
            plotdata = np.zeros((len(position), len(frequency)))
            plotflags = np.zeros((len(position), len(frequency)), dtype=bool)
            for i, idx in enumerate(range(0, data.shape[0], skiptime)):
                print(idx, end="\r")
                plotflags[i] = flag[i, ::skipchan]
                if datatype == "XX":
                    plotdata[i] = data[idx, 0, ::skipchan]
                elif datatype == "YY":
                    plotdata[i] = data[idx, 1, ::skipchan]
                elif datatype == "Re(XY)":
                    plotdata[i] = data[idx, 2, ::skipchan]
                elif datatype == "Im(XY)":
                    plotdata[i] = data[idx, 3, ::skipchan]
                else:
                    raise ValueError(f"Invalid datatype: {datatype}")

            if not plotcal:
                # remove cal
                plotdata[cal] = np.nan

            if not plotflagged:
                # apply flags
                plotdata[plotflags] = np.nan

            # Plot
            print(f"Plotting {datatype}")
            vmin, vmax = np.nanpercentile(plotdata, q=[5.0, 95.0])
            fig, ax = plt.subplots(figsize=(12, 12))
            cax = ax.imshow(
                plotdata,
                extent=extent,
                interpolation="none",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(cax, ax=ax, label="Relative Power")
            ax.set_xlabel(f"{freqtype} Frequency (MHz)")
            ax.set_ylabel(f"Seconds Since MJD = {position['MJD'][0]:.6f}")
            ax.set_title(f"{datafile} {datatype}")
            fig.tight_layout()
            fname = f"{prefix}_{label}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate waterfall figures from SDHDF file",
        prog="waterfall.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datafile", type=str, help="SDHDF file",
    )
    parser.add_argument(
        "prefix", type=str, help="Output plot files prefix",
    )
    parser.add_argument(
        "-t", "--skiptime", type=int, default=1, help="Plot every n-th integration"
    )
    parser.add_argument(
        "-c", "--skipchan", type=int, default=1, help="Plot every n-th channel"
    )
    parser.add_argument(
        "--plotcal",
        action="store_true",
        default=False,
        help="Plot cal-on integrations",
    )
    parser.add_argument(
        "--plotflagged", action="store_true", default=False, help="Plot flagged data",
    )
    args = parser.parse_args()
    waterfall(
        args.datafile,
        args.prefix,
        skiptime=args.skiptime,
        skipchan=args.skipchan,
        plotcal=args.plotcal,
        plotflagged=args.plotflagged,
    )


if __name__ == "__main__":
    main()
