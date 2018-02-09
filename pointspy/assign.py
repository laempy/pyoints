import numpy as np
from scipy.spatial import cKDTree
from IndexKD import IndexKD


def pairs(aCoords, bCoords, distance_upper_bound=np.inf):

    dtype = [('A', int), ('B', int), ('weights', float)]

    aIndexKD = IndexKD(aCoords)
    aDists, aIds = aIndexKD.kNN(
        bCoords, k=1, distance_upper_bound=distance_upper_bound)

    bIndexKD = IndexKD(bCoords)
    bDists, bIds = bIndexKD.kNN(
        aCoords, k=1, distance_upper_bound=distance_upper_bound)

    pairs = []
    for aId in range(len(aIds)):
        if aDists[aId] < distance_upper_bound:
            if aId == bIds[aIds[aId]]:
                pairs.append((aIds[aId], aId))

    pairs = np.array(pairs, dtype=int)

    #print aDists[pairs[:,1]]
    #print bDists[pairs[:,0]]

    return pairs


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
    print w
    print w.max()
    print w.min()
    print w.sum(1)
    print aCoords
    print aCoords[keys, :]
    mCoords = (aCoords[keys, :].T * w).T.sum(0)
    # mCoords=aCoords[keys,:].mean(0)
    print mCoords
    print mCoords.shape

    return
    exit(0)
    for i in range(k):

        print aCoords[aIds[:, i], :]

    pairs = []
    for aId in range(len(aIds)):
        if aDists[aId] < distance_upper_bound:
            if aId == bIds[aIds[aId]]:
                pairs.append((aIds[aId], aId))
    pairs = np.array(pairs, dtype=int)

    #print aDists[pairs[:,1]]
    #print bDists[pairs[:,0]]

    return pairs
