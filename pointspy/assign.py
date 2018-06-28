"""Module to find pairs of points"""

import numpy as np

from . import (
    assertion,
    transformation,
    IndexKD,
)


class Pairs:

    def __init__(self, coords, radii):

        # validate input
        coords = assertion.ensure_coords(coords)
        radii = assertion.ensure_numvector(radii, length=coords.shape[1])

        S = transformation.s_matrix(1.0 / radii)
        self.rIndexKD = IndexKD(coords, S)

    def __call__(self, coords):

        mIndexKD = IndexKD(coords, self.rIndexKD.t)
        rIndexKD = self.rIndexKD

        rDists, rIds = rIndexKD.knn(
            mIndexKD.coords, k=1, distance_upper_bound=1)

        mDists, mIds = mIndexKD.knn(
            rIndexKD.coords, k=1, distance_upper_bound=1)

        pairs = []
        for rId in range(len(rIds)):
            if rDists[rId] <= 1:
                if rId == mIds[rIds[rId]]:
                    pairs.append((rIds[rId], rId))

        return np.array(pairs, dtype=int)



class BallAssign:


    def __init__(self, coords, radii):

        # validate input
        coords = assertion.ensure_coords(coords)
        radii = assertion.ensure_numvector(radii, length=coords.shape[1])

        S = transformation.s_matrix(1.0 / radii)
        self.rIndexKD = IndexKD(coords, S)


    def __call__(self, coords):
        mIndexKD = IndexKD(coords, self.rIndexKD.t)
        rIndexKD = self.rIndexKD
        
        pairs = []
        ball_gen = rIndexKD.ball_iter(mIndexKD.coords, 1)
        for mId, rIds in enumerate(ball_gen):
            for rId in rIds:
                pairs.append((rId, mId))
                
        return np.array(pairs, dtype=int)



# TODO class Pairs ==> indexKD reuse
def pairs(aCoords, bCoords, max_distance=np.inf):
    """Find pairs of points using nearest neighbour method.

    Parameters
    ----------
    aCoords : array_like(Number, shape=(n, k))
        Represents `n` data points of `k` dimensions.
    bCoords : array_like(Number, shape=(m, k))
        Represents `m` data points of `k` dimensions.
    max_distance : optional, positive float
        Maximum distance to assign a point pair.

    Returns
    -------
    pairs : np.ndarray(int, shape=(p, 2))
        Indices of identified pairs. For each row `(a, b)` in `pairs`
        `aCoords[a, :]` is assigned to `bCoords[b, :]`.

    Examples
    --------

    >>> aCoords = [(0.2, 0), (1.2, 1), (2, 1), (3.2, 4), (-2, -4)]
    >>> bCoords = [(0.1, 0), (2, 1), (1, 1.1), (3.5, 4), (2.5, -4)]
    >>> print pairs(aCoords, bCoords, max_distance=0.5)
    ds

    """

    aIndexKD = IndexKD(aCoords)
    aDists, aIds = aIndexKD.knn(
        bCoords, k=1, distance_upper_bound=max_distance)

    bIndexKD = IndexKD(bCoords)
    bDists, bIds = bIndexKD.knn(
        aCoords, k=1, distance_upper_bound=max_distance)

    pairs = []
    for aId in range(len(aIds)):
        if aDists[aId] < max_distance:
            if aId == bIds[aIds[aId]]:
                pairs.append((aIds[aId], aId))

    return np.array(pairs, dtype=int)


def kNN(aCoords, bCoords, dim=None, k=2, distance_upper_bound=np.inf):

    if dim is None:
        dim = aCoords.shape[0] - 1

    indexKD = IndexKD(aCoords[:, 0:dim])
    dists, nIds = indexKD.kNN(
        bCoords[:, 0:dim], k=k, distance_upper_bound=distance_upper_bound)

    mask = np.all(dists < distance_upper_bound, axis=1)
    nIds = nIds[mask, :]
    dists = dists[mask, :]

    keys = zip(*nIds)
    w = dists / np.repeat(dists.sum(1), k).reshape((len(dists), k))

    # TODO
    #print w
    #print w.max()
    #print w.min()
    #print w.sum(1)
    #print aCoords
    #print aCoords[keys, :]
    mCoords = (aCoords[keys, :].T * w).T.sum(0)
    # mCoords=aCoords[keys,:].mean(0)
    #print mCoords
    #print mCoords.shape

    return
    exit(0)
    #for i in range(k):

    pairs = []
    for aId in range(len(aIds)):
        if aDists[aId] < distance_upper_bound:
            if aId == bIds[aIds[aId]]:
                pairs.append((aIds[aId], aId))
    pairs = np.array(pairs, dtype=int)

    #print aDists[pairs[:,1]]
    #print bDists[pairs[:,0]]

    return pairs
