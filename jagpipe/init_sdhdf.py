"""
init_sdhdf.py
Initialize a new JAG SDHDF data file.

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

from astropy.time import Time
import numpy as np


def init_sdhdf(sdhdf, freqaxis, freqframe, exposure):
    """
    Initialize a new SDHDF data structure assuming single beam, single
    band, single scan.

    Inputs:
        sdhdf :: h5py.File
            SDHDF file handle
        freqaxis :: 1-D array of scalars
            Frequency axis (Hz)
        freqframe :: string
            Frequency axis frame
        exposure :: scalar
            Integration time (s)

    Returns: metadata, data, position, flag
        metadata :: h5py.Group
            Metadata group
        data, position, flag :: h5py.Dataset
            Datasets for data, position, and flag
    """
    # Setup file structure
    metadata = sdhdf.create_group("metadata")
    metadata.attrs["NAME"] = "metadata"
    metadata.attrs["DESCRIPTION"] = "Observation metadata"
    metadata.attrs["SDHDF_VERSION"] = "v2.0-alpha20210923"
    metadata.attrs["OBSERVER"] = "Wenger"
    metadata.attrs["TELESCOPE"] = "DRAO JAG 26m"
    metadata.attrs["PROJECTID"] = "DRAO-JAG-21B-000"
    metadata.attrs["RECEIVER"] = "CHIME"
    metadata.attrs["BACKEND"] = "Backend Name"
    metadata.attrs["DATE"] = Time.now().isot

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
    frequency.attrs["FRAME"] = freqframe
    frequency.attrs["UNIT"] = "Hz"

    # Data - Beam 0 - Band SB0 - Polarization
    polarization_data = np.array(["XX", "YY", "Re(XY)", "Im(XY)"], dtype="S")
    polarization = band_sb0.create_dataset(
        "polarization", data=polarization_data
    )
    polarization.attrs["NAME"] = "polarization"
    polarization.attrs["DESCRIPTION"] = "Polarization types"

    # Data - Beam 0 - Band SB0 - Scan 0
    scan = band_sb0.create_group("scan_0")
    scan.attrs["NAME"] = "scan_0"
    scan.attrs["DESCRIPTION"] = "Data specific to scan 0"
    scan.attrs["TYPE"] = "ON"

    # Metadata storage
    position_data = np.array(
        [],
        dtype=[
            ("SOURCE", "S20"),
            ("UTC", "S23"),
            ("MJD", "f8"),
            ("AZIMUTH", "f8"),
            ("ELEVATION", "f8"),
            ("RA", "f8"),
            ("DEC", "f8"),
        ],
    )
    scan_position = scan.create_dataset(
        "position",
        data=position_data,
        maxshape=(None,),
    )
    scan_position.attrs["NAME"] = "position"
    scan_position.attrs["DESCRIPTION"] = "Position at start of integration"
    scan_position.attrs["COLUMN_FRAMES"] = [
        "Source",
        "UTC",
        "MJD",
        "",
        "",
        "J2000",
        "J2000",
    ]
    scan_position.attrs["COLUMN_UNITS"] = [
        "",
        "",
        "",
        "DEG",
        "DEG",
        "DEG",
        "DEG",
    ]

    # Chunk shape for data storage
    # For float, this is about 12 MB
    chunk_time = 20
    chunk_pol = 4
    chunk_freq = min(20000, len(freqaxis))
    chunks = (chunk_time, chunk_pol, chunk_freq)

    # Data storage
    data = np.empty((0, 4, len(freqaxis)), dtype=float)

    scan_data = scan.create_dataset(
        "data",
        data=data,
        maxshape=(None, 4, len(freqaxis)),
        chunks=chunks,
    )
    scan_data.attrs["NAME"] = "data"
    scan_data.attrs["DESCRIPTION"] = "Data"
    scan_data.attrs["DIMENSION_LABELS"] = ["time", "pol", "channel"]
    scan_data.attrs["TYPE"] = "Antenna Temperature"
    scan_data.attrs["UNIT"] = "K"

    # Flag storage
    data = np.empty((0, len(freqaxis)), dtype=bool)
    chunks = (chunk_time, chunk_freq)
    scan_flag = scan.create_dataset(
        "flag",
        data=data,
        maxshape=(None, len(freqaxis)),
        chunks=chunks,
    )
    scan_flag.attrs["NAME"] = "flag"
    scan_flag.attrs["DESCRIPTION"] = "Flag"
    scan_flag.attrs["DIMENSION_LABELS"] = ["time", "channel"]

    # Cal mask storage
    data = np.empty((0), dtype=bool)
    scan_cal = scan.create_dataset(
        "cal",
        data=data,
        maxshape=(None,),
    )
    scan_cal.attrs["NAME"] = "cal"
    scan_cal.attrs["DESCRIPTION"] = "Cal"
    scan_cal.attrs["DIMENSION_LABELS"] = ["time"]

    return metadata, scan_data, scan_position, scan_flag
