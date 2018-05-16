"""Smoothing of point clouds.
"""
import numpy as np

from .indexkd import IndexKD
from . import assertion


def mean_ball(coords, r, numIter=1, updatePairs=False):
    """Smoothing of spatial structures by averaging neighboured point
    coordinates.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Array representing `n` points with `k` dimensions.
    r : Number
        Maximum distance to nearby points used to average the coordinates.
    numIter : optional, positive int
        Number of iterations.
    updatePairs : optional, bool
        Specifies weather or not point pairs are updated on each iteration.

    See Also
    --------
    mean_knn

    """
    coords = assertion.ensure_coords(coords)
    if not assertion.isnumeric(r):
        raise ValueError("'r' needs to a number")
    if not (isinstance(numIter, int) and numIter > 0):
        raise ValueError("'numIter' needs to be an integer greater zero")
    if not isinstance(updatePairs, bool):
        raise ValueError("'updatePairs' needs to be boolean")

    ids = None
    mCoords = np.copy(coords)
    for _ in range(numIter):

        if ids is None or updatePairs:
            indexKD = IndexKD(mCoords)
            ids = indexKD.ball(indexKD.coords(), r)

        # averaging
        mCoords = np.array([mCoords[nIds, :].mean(0) for nIds in ids])

    return mCoords


def mean_knn(coords, k, numIter=1, updatePairs=False):
    """Smoothing of spatial structures by averaging neighboured point
    coordinates.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, l))
        Array representing `n` points with `l` dimensions.
    k : float
        Number of nearest points used to average the coordinates.
    numIter : optional, int
        Number of iterations.
    updatePairs : optional, bool
        Specifies weather or not point pairs are updated on each iteration.

    See Also
    --------
    mean_ball

    """
    coords = assertion.ensure_coords(coords)
    if not (isinstance(k, int) and k > 0):
        raise ValueError("'k' needs to be an integer greater zero")
    if not (isinstance(numIter, int) and numIter > 0):
        raise ValueError("'numIter' needs to be an integer greater zero")
    if not isinstance(updatePairs, bool):
        raise ValueError("'updatePairs' needs to be boolean")

    ids = None
    mCoords = np.copy(coords)
    for _ in range(numIter):

        if ids is None or updatePairs:
            indexKD = IndexKD(mCoords)
            ids = indexKD.kNN(indexKD.coords(), k=k)[1]

        # averaging
        mCoords = np.array([mCoords[nIds, :].mean(0) for nIds in ids])

    return mCoords
