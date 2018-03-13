import numpy as np
from .indexkd import IndexKD
from .interpolate import LinearInterpolator


# TODO density (min neighbours in radius)


def extrema(indexKD, attributes, r=1, inverse=False):

    coords = indexKD.coords
    order = np.argsort(attributes)
    notClassified = np.ones(len(order), dtype=np.bool)

    if inverse:
        def check(id, nIds):
            value = attributes[id]
            for nId in nIds:
                if attributes[nId] < value:
                    return False
            return True
    else:
        def check(id, nIds):
            value = attributes[id]
            for nId in nIds:
                if attributes[nId] > value:
                    return False
            return True
        order = order[::-1]

    for id in order:
        if notClassified[id]:
            nIds = indexKD.ball(coords[id, :], r)
            notClassified[nIds] = False
            if check(id, nIds):
                yield id


def hasNeighbour(indexKD, r=1):
    coords = indexKD.coords

    notClassified = np.ones(len(indexKD), dtype=np.bool)
    for pId, coord in enumerate(coords):
        if notClassified[pId]:
            nIds = np.array(indexKD.ball(coord, r))
            if len(nIds) > 1:
                notClassified[nIds] = False
            else:
                yield pId


def ball(indexKD, r=1, order=None, inverse=False, axis=-1, min_pts=1):
    coords = indexKD.coords

    if order is None:
        order = np.argsort(coords[:, axis])[::-1]

    if inverse:
        order = order[::-1]

    if not hasattr(r, '__len__'):
        r = np.repeat(r, len(indexKD))

    notClassified = np.ones(len(order), dtype=np.bool)
    for pId in order:
        if notClassified[pId]:
            nIds = indexKD.ball(coords[pId], r[pId])
            if len(nIds) >= min_pts:
                notClassified[nIds] = False
                yield pId


def surface(indexKD, r=1, order=None, inverse=False, axis=-1):

    coords = indexKD.coords

    if order is None:
        order = np.argsort(coords[:, axis])[::-1]

    if inverse:
        order = order[::-1]
    inverseOrder = np.argsort(order)

    notClassified = np.ones(len(order), dtype=np.bool)
    for pId in order:
        if notClassified[pId]:
            nIds = np.array(indexKD.ball(coords[pId, :], r))
            notClassified[nIds] = False
            yield nIds[np.argmin(inverseOrder[nIds])]


def inConvexHull(hullCoords, coords):
    interpolator = LinearInterpolator(
        hullCoords, np.zeros(
            hullCoords.shape[0]))
    return ~np.isnan(interpolator(coords))


def minFilter(indexKD, r, axis=-1):
    coords = indexKD.coords

    ballIter = indexKD.ball(coords, r)
    mask = np.zeros(len(indexKD), dtype=bool)
    for nIds in ballIter:
        nId = nIds[np.argmin(coords[nIds, axis])]
        mask[nId] = True
    return mask.nonzero()


def dem(coords, r=1, order=None, inverse=False, axis=-1):
    dim = coords.shape[1]
    assert dim >= 3

    if order is None:
        order = np.argsort(coords[:, axis])
    if inverse:
        order = order[::-1]
    if not hasattr(r, '__getitem__'):
        r = np.repeat(r, len(order))

    # filter
    ballGen = ball(IndexKD(coords[:, :-1]), r, order=order)
    fIds = np.array(list(ballGen), dtype=int)

    # ensure neighbours
    count = IndexKD(coords[fIds, :]).ball_count(2 * r[fIds])
    fIds = fIds[count >= 6]

    # subsequent filter
    while True:
        l = len(fIds)
        count = IndexKD(coords[fIds, :]).ball_count(2 * r[fIds])
        fIds = fIds[count >= 3]
        if l == len(fIds):
            break

    return fIds
