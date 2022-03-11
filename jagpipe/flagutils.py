"""
flagutils.py
Utility functions for data flagging.

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

import numpy as np

_MAX_INT = np.iinfo(int).max


def gradient_filter(data, delta=1, cutoff=5.0):
    """
    Use a first and second order finite differences filter along the
    first axis of some data to identify narrow interference spikes.
    N.B. Data must be positive definite.

    Inputs:
        data :: N-D array of scalars
            Data. Filter is applied along first axis
        delta :: integer
            Finite difference window size
        cutoff :: scalar
            Sigma-clipping threshold

    Returns: mask
        mask :: N-D array of boolean
            Mask (True = bad)
    """
    # First order
    foo = np.roll(data, -1 * delta, axis=0)
    bar = np.roll(data, +1 * delta, axis=0)
    grad = (bar - foo) / data
    med_grad = np.nanmedian(grad, axis=0)
    rms_grad = 1.4826 * np.nanmedian(np.abs(grad - med_grad), axis=0)
    mask = np.abs(grad - med_grad) > cutoff * rms_grad
    # second order
    grad = (bar - 2.0 * data + foo) / data
    med_grad = np.nanmedian(grad, axis=0)
    rms_grad = 1.4826 * np.nanmedian(np.abs(grad - med_grad), axis=0)
    mask = mask | (np.abs(grad - med_grad) > cutoff * rms_grad)
    return mask


def rolling_sum(data, window):
    """
    Calculate rolling sum along the first axis of some data.
    Edges are handled by reflection.

    Inputs:
        data :: N-D array of scalars
            Data. Rolling sum is applied along first axis.
        window :: odd integer
            Window width

    Returns:
        total :: N-D array of scalars
            Rolling sum
        notnan :: N-D array of scalars
            Number of non-nan elements included in rolling sum
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    halfwin = (window - 1) // 2
    newdata = np.concatenate([data[:halfwin][::-1], data, data[-halfwin:][::-1]])
    nansum = np.cumsum(~np.isnan(newdata), axis=0)
    cumsum = np.nancumsum(newdata, axis=0)
    # append with zero
    total = (
        cumsum[window - 1 :]
        - np.r_[np.zeros((1,) + cumsum.shape[1:]), cumsum[:-window]]
    )
    notnan = (
        nansum[window - 1 :]
        - np.r_[np.zeros((1,) + nansum.shape[1:]), nansum[:-window]]
    )
    notnan[notnan < 2] = _MAX_INT  # all nan
    return total, notnan


def rolling_mean(data, window):
    """
    Calculate rolling mean along the first axis of some data.
    Edges are handled by reflection.

    Inputs:
        data :: N-D array of scalars
            Data. Mean is taken along first axis
        window :: odd integer
            Window width

    Returns:
        mean :: N-D array of scalars
            Rolling mean
    """
    total, notnan = rolling_sum(data, window)
    return total / notnan


def rolling_std(data, window):
    """
    Calculate rolling standard deviation along the first axis of some data.
    Edges are handled by reflection.
    N.B. This is an approximation with ~0.1% accuracy.

    Inputs:
        data :: N-D array of scalars
            Data. Standard deviation is taken along first axis.
        window :: odd integer
            Window width

    Returns:
        std :: N-D array of scalars
            Rolling standard deviation
    """
    mean = rolling_mean(data, window)
    total, notnan = rolling_sum((data - mean) ** 2.0, window)
    return np.sqrt(total / (notnan - 1))


def generate_flag_mask(data, mask, window=101, cutoff=5.0, grow=0.75):
    """
    Automatically flag a spectrum using
    1. gradient mask (narrow interference)
    2. sigma clipping (narrow inteference)
    3. rms clipping (broad interference)
    4. mask expansion (broad, low-level interference)
    All filters are applied along the first axis.

    Inputs:
        data :: N-D array of scalars (shape: A x 4 x ...)
            A-length data, with [XX, YY, Re(XY), Im(XY)]
            along the second axis. Filters are applied along first axis.
        mask :: (N-1)-D array of boolean (shape: A x ...)
            Initial boolean mask
        window :: integer
            Rolling window size along first axis
        cutoff :: scalar
            Sigma clip
        grow :: scalar between 0.0 and 1.0
            Extend mask where more than grow*window adjacent data
            are masked

    Returns:
        new_mask :: (N-1)-D array of boolean (shape: A x ...)
            Updated boolean mask
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")

    # compute total power and cross power
    total_power = data[:, 0] + data[:, 1]
    cross_power = np.sqrt(2.0) * (data[:, 2] + data[:, 3])

    # interpolate through NaNs along the first axis
    x = np.arange(total_power.shape[0])
    for i in range(total_power.shape[1]):
        isnan = np.isnan(total_power[:, i])
        if np.all(isnan):
            continue
        total_power[isnan, i] = np.interp(x[isnan], x[~isnan], total_power[~isnan, i])
        cross_power[isnan, i] = np.interp(x[isnan], x[~isnan], cross_power[~isnan, i])

    # flag narrow spikes in total power using gradient filter
    gradient_mask = gradient_filter(total_power, delta=1, cutoff=cutoff)
    mask = mask | gradient_mask

    # update mask
    total_power[mask] = np.nan
    cross_power[mask] = np.nan

    # rolling window mean on total power
    mean_tp = rolling_mean(total_power, window)

    # rolling window mean and rms on cross correlations.
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
    med = np.nanmedian(norm_rms, axis=0)
    rms = 1.4826 * np.nanmedian(np.abs(norm_rms - med), axis=0)
    rms[rms == 0.0] = np.nan
    norm_rms = (norm_rms - med) / rms
    mask = mask | (np.abs(norm_rms) > cutoff)

    # flag channels where >grow fraction of adjacent channels are flagged
    flagged = rolling_mean(mask, window)
    mask = mask | (flagged > grow)
    return mask
