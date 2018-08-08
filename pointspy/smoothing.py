# BEGIN OF LICENSE NOTE
# This file is part of PoYnts.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Smoothing of point clouds.
"""

import numpy as np

from .indexkd import IndexKD
from . import assertion


def mean_ball(coords, r, num_iter=1, update_pairs=False):
    """Smoothing of spatial structures by averaging neighboured point
    coordinates.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Array representing `n` points with `k` dimensions.
    r : Number
        Maximum distance to nearby points used to average the coordinates.
    num_iter : optional, positive int
        Number of iterations.
    update_pairs : optional, bool
        Specifies weather or not point pairs are updated on each iteration.

    See Also
    --------
    mean_knn

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
        mCoords = np.array([mCoords[nIds, :].mean(0) for nIds in ids])

    return mCoords


def mean_knn(coords, k, num_iter=1, update_pairs=False):
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

    See Also
    --------
    mean_ball

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
            ids = indexKD.kNN(indexKD.coords(), k=k)[1]

        # averaging
        mCoords = np.array([mCoords[nIds, :].mean(0) for nIds in ids])

    return mCoords
