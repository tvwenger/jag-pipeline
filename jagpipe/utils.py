"""
utils.py
General utilities.

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

import numpy as np
from astropy.time import Time


def add_history(sdhdf, message):
    """
    Add an item to the history of an SDHDF file.

    Inputs:
        sdhdf :: h5py.File
            SDHDF file handle
        message :: string
            Message

    Returns: Nothing
    """
    history = sdhdf["metadata"]["history"]
    history.resize(history.shape[0] + 1, axis=0)
    row = np.array((Time.now().isot, message), dtype=history.dtype)
    history[-1] = row
