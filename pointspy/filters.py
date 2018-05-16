"""Point filters.
"""

import numpy as np

from . import (
    assertion,
    interpolate,
    IndexKD,
)


# TODO density (min neighbours in radius)

def extrema(indexKD, attributes, r=1, inverse=False):

    coords = indexKD.coords
    order = np.argsort(attributes)
    not_classified = np.ones(len(order), dtype=np.bool)

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
        if not_classified[id]:
            nIds = indexKD.ball(coords[id, :], r)
            not_classified[nIds] = False
            if check(id, nIds):
                yield id


def has_neighbour(indexKD, r):
    """Decide whether or not points have neighbouring points.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index with points to analyze.
    r : positive Number
        Maximum distance of a point to a neighbouring point to be still
        considered as isolated.

    Yields
    ------
    positive int
        Yields the index of each point with at least a neighbouring point with
        radius less or equal `r`.

    See also
    --------
    is_isolated

    Examples
    --------

    >>> coords = [(0, 0), (0.5, 0.5), (0, 1), (0.7, 0.5), (-1, -1)]
    >>> indexKD = IndexKD(coords)
    >>> print list(has_neighbour(indexKD, 0.7))
    [1, 3]

    """
    if not isinstance(indexKD, IndexKD):
        raise ValueError("'indexKD' needs to be an instance of IndexKD")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs be a number greater zero")

    not_classified = np.ones(len(indexKD), dtype=np.bool)
    for pId, coord in enumerate(indexKD.coords):
        if not_classified[pId]:
            nIds = np.array(indexKD.ball(coord, r))
            if len(nIds) > 1:
                not_classified[nIds] = False
                yield pId
        else:
            yield pId


def is_isolated(indexKD, r):
    """Decide whether or not points have neighbouring points.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index with points to analyze.
    r : positive Number
        Maximum distance of a point to a neighbouring point to be still
        considered as isolated.

    Yields
    ------
    positive int
        Yields the index of each isolated point.

    See Also
    --------
    has_neighbour

    Examples
    --------

    >>> coords = [(0, 0), (0.5, 0.5), (0, 1), (0.7, 0.5), (-1, -1)]
    >>> indexKD = IndexKD(coords)
    >>> print list(is_isolated(indexKD, 0.7))
    [0, 2, 4]

    """
    if not isinstance(indexKD, IndexKD):
        raise ValueError("'indexKD' needs to be an instance of IndexKD")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs be a number greater zero")

    not_classified = np.ones(len(indexKD), dtype=np.bool)
    for pId, coord in enumerate(indexKD.coords):
        if not_classified[pId]:
            nIds = np.array(indexKD.ball(coord, r))
            if len(nIds) > 1:
                not_classified[nIds] = False
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

    not_classified = np.ones(len(order), dtype=np.bool)
    for pId in order:
        if not_classified[pId]:
            nIds = indexKD.ball(coords[pId], r[pId])
            if len(nIds) >= min_pts:
                not_classified[nIds] = False
                yield pId


def in_convex_hull(hull_coords, coords):
    """Decide whether or not points are within a convex hull.

    Parameters
    ----------
    hull_coords : array_like(Number, shape=(m, k))
        Represents `m` hull points of `k` dimensions.
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points to check whether or not they are located
        within the convex hull.

    Returns
    -------
    array_like(Bool, shape=(n))
        Indicates whether or not the points are located within the convex hull.

    Examples
    --------

    >>> hull = [(0, 0), (1, 0), (1, 2)]
    >>> coords = [(0, 0), (0.5, 0.5), (0, 1), (0.7, 0.5), (-1, -1)]
    >>> print in_convex_hull(hull, coords)
    [ True  True False  True False]

    """
    hull_coords = assertion.ensure_coords(hull_coords)
    coords = assertion.ensure_coords(coords)

    interpolator = interpolate.LinearInterpolator(
            hull_coords,
            np.zeros(hull_coords.shape[0])
        )
    return ~np.isnan(interpolator(coords))


def minFilter(indexKD, r, axis=-1):
    coords = indexKD.coords

    ballIter = indexKD.ball_iter(coords, r, bulk = 1000)
    mask = np.zeros(len(indexKD), dtype=bool)
    for nIds in ballIter:
        nId = nIds[np.argmin(coords[nIds, axis])]
        print nId
        mask[nId] = True
    return mask.nonzero()


# TODO unterschied zu dem
def surface(indexKD, r=1, order=None, inverse=False, axis=-1):

    coords = indexKD.coords

    if order is None:
        order = np.argsort(coords[:, axis])[::-1]

    if inverse:
        order = order[::-1]
    inverseOrder = np.argsort(order)

    not_classified = np.ones(len(order), dtype=np.bool)
    for pId in order:
        if not_classified[pId]:
            nIds = np.array(indexKD.ball(coords[pId, :], r))
            not_classified[nIds] = False
            yield nIds[np.argmin(inverseOrder[nIds])]

# TODO unterschied zu surface
def dem(coords, r, order=None, inverse=False, axis=-1):
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
