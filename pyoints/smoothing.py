# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, Trier University, 
# lamprecht@uni-trier.de
# 
# Pyoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Pyoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# END OF LICENSE NOTE
"""Collection of algorithms to smooth point clouds.
"""

import numpy as np

from .indexkd import IndexKD
from . import assertion


def mean_ball(
    coords, r,
    num_iter=1,
    update_pairs=False,
    f=lambda coord, ncoords: ncoords.mean(0)
):
    """Smoothing of spatial structures by iterative averaging the coordinates
    of neighboured points.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Array representing `n` points of `k` dimensions.
    r : Number
        Maximum distance to nearby points used to average the coordinates.
    num_iter : optional, positive int
        Number of iterations.
    update_pairs : optional, bool
        Specifies weather or not point pairs are updated on each iteration.
    f : callable
        Aggregate function used for smoothing. It recieves the original point
        coordinate and the coordinates of neighboured points as an argument
        and returns a smoothed coordinate.

    See Also
    --------
    mean_knn

    Examples
    --------

    Create a three dimensional irregular suface of points.

    >>> coords = np.ones((100, 3), dtype=float)
    >>> coords[:, 0:2] = np.vstack(np.mgrid[0:10, 0:10].T)
    >>> coords[:, 2] = np.tile([1.05, 0.95], 50)

    Get value range in each coordinate dimension.

    >>> print(np.ptp(coords, axis=0))
    [9.  9.  0.1]

    Smooth coordinates to get a more regular surface. But, the first two
    coordinate dimensions are affected, too.

    >>> scoords = mean_ball(coords, 1.5)
    >>> print(np.round(np.ptp(scoords, axis=0), 3))
    [8.    8.    0.033]

    Modyify the aggregation function to smooth the third coordinate axis only.

    >>> def aggregate_function(coord, ncoords):
    ...     coord[2] = ncoords[:, 2].mean(0)
    ...     return coord
    >>> scoords = mean_ball(coords, 1.5, f=aggregate_function)
    >>> print(np.round(np.ptp(scoords, axis=0), 3))
    [9.    9.    0.026]

    Increase number of iterations to get a smoother result.

    >>> scoords = mean_ball(coords, 1.5, num_iter=3, f=aggregate_function)
    >>> print(np.round(np.ptp(scoords, axis=0), 3))
    [9.   9.   0.01]

    """
    coords = assertion.ensure_coords(coords)
    if not assertion.isnumeric(r):
        raise TypeError("'r' needs to a number")
    if not (isinstance(num_iter, int) and num_iter > 0):
        raise ValueError("'num_iter' needs to be an integer greater zero")
    if not isinstance(update_pairs, bool):
        raise TypeError("'update_pairs' needs to be boolean")

    ids = None
    mCoords = np.copy(coords)
    for _ in range(num_iter):

        if ids is None or update_pairs:
            indexKD = IndexKD(mCoords)
            ids = indexKD.ball(indexKD.coords, r)

        # averaging
        mCoords = np.array([
            f(mCoords[i, :], mCoords[nIds, :]) for i, nIds in enumerate(ids)
        ])

    return mCoords


def mean_knn(
    coords,
    k,
    num_iter=1,
    update_pairs=False,
    f=lambda coord, ncoords: ncoords.mean(0)
):
    """Smoothing of spatial structures by averaging neighboured point
    coordinates.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, l))
        Array representing `n` points with `l` dimensions.
    k : float
        Number of nearest points used to average the coordinates.
    num_iter : optional, int
        Number of iterations.
    update_pairs : optional, bool
        Specifies weather or not point pairs are updated on each iteration.
    f : callable
        Aggregate function used for smoothing. It recieves the original point
        coordinate and the coordinates of neighboured points as an argument
        and returns a smoothed coordinate.

    See Also
    --------
    mean_ball

    Examples
    --------

    Create a three dimensional irregular suface of points.

    >>> coords = np.ones((100, 3), dtype=float)
    >>> coords[:, 0:2] = np.vstack(np.mgrid[0:10, 0:10].T)
    >>> coords[:, 2] = np.tile([1.05, 0.95], 50)

    Get value range in each coordinate dimension.

    >>> print(np.ptp(coords, axis=0))
    [9.  9.  0.1]

    Smooth coordinates to get a more regular surface. But, the first two
    coordinate dimensions are affected, too.

    >>> scoords = mean_knn(coords, 5)
    >>> print(np.round(np.ptp(scoords, axis=0), 3))
    [8.2  8.2  0.02]

    Modyify the aggregation function to smooth the third coordinate axis only.

    >>> def aggregate_function(coord, ncoords):
    ...     coord[2] = ncoords[:, 2].mean(0)
    ...     return coord
    >>> scoords = mean_knn(coords, 5, f=aggregate_function)
    >>> print(np.round(np.ptp(scoords, axis=0), 3))
    [9.    9.    0.033]

    """
    coords = assertion.ensure_coords(coords)
    if not (isinstance(k, int) and k > 0):
        raise ValueError("'k' needs to be an integer greater zero")
    if not (isinstance(num_iter, int) and num_iter > 0):
        raise ValueError("'num_iter' needs to be an integer greater zero")
    if not isinstance(update_pairs, bool):
        raise TypeError("'update_pairs' needs to be boolean")

    ids = None
    mCoords = np.copy(coords)
    for _ in range(num_iter):

        if ids is None or update_pairs:
            indexKD = IndexKD(mCoords)
            ids = indexKD.knn(indexKD.coords, k=k)[1]

        # averaging
        mCoords = np.array([
            f(mCoords[i, :], mCoords[nIds, :]) for i, nIds in enumerate(ids)
        ])

    return mCoords
