"""Point filters.
"""

import numpy as np

from . import (
    assertion,
    distance,
    interpolate,
    IndexKD,
)


# TODO density (min neighbours in radius)
def extrema(indexKD, attributes, r, inverse=False):
    """Find local maxima or minima for a given point set.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index of `n` points to filter.
    attributes : array_like(Number, shape=(n))
        Attributes to search for extrema. If None the last coordinate dimension
        is used.
    r : positive Number
        Maximum distance between two points to be considered as neighbours.
    inverse : optional, bool
        Indicates if local maxima (False) or local minima (True) shall be
        identfied.

    Yields
    ------
    positive int
        Indices of local maxima or minima.

    Examples
    --------

    Local maxima.

    >>> indexKD = IndexKD([(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5) ])
    >>> attributes = [2, 0.1, 1, 0, 0.5]
    >>> fIds = list(extrema(indexKD, attributes, 1.1))
    >>> print(fIds)
    [0, 2]

    Local minima.

    >>> fIds = list(extrema(indexKD, attributes, 1.1, inverse=True))
    >>> print(fIds)
    [3, 1]

    Just one local minimum.

    >>> fIds = list(extrema(indexKD, attributes, 1.5, inverse=True))
    >>> print(fIds)
    [3]

    """

    # type validation

    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be an instance of 'IndexKD'")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs be a number greater zero")

    attributes = assertion.ensure_numvector(
        attributes,
        min_length=len(indexKD),
        max_length=len(indexKD)
    )
    if not inverse:
        attributes = -attributes

    coords = indexKD.coords
    order = np.argsort(attributes)
    not_classified = np.ones(len(order), dtype=np.bool)

    # define filter function
    def check(pId, nIds):
        value = attributes[pId]
        for nId in nIds:
            if attributes[nId] < value:
                return False
        return True

    # filtering
    for pId in order:
        if not_classified[pId]:
            nIds = indexKD.ball(coords[pId, :], r)
            not_classified[nIds] = False
            if check(pId, nIds):
                yield pId


def min_filter(indexKD, attributes, r, inverse=False):
    """Find minima or maxima within a specified radius.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index of `n` points to filter.
    attributes : array_like(Number, shape=(n))
        Attributes to search for extrema. If None the last coordinate dimension
        is used.
    r : positive Number
        Local radius to search for a minimum.
    inverse : optional, bool
        Indicates if local minima (False) or local maxima (True) shall be
        identfied.

    Returns
    -------
    np.ndarray(int, shape=(n))
        Filtered point indices.

    """

    # type validation
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs be a number greater zero")

    attributes = assertion.ensure_numvector(
        attributes,
        min_length=len(indexKD),
        max_length=len(indexKD)
    )
    if inverse:
        attributes = -attributes

    # filtering
    ballIter = indexKD.ball_iter(indexKD.coords, r, bulk=1000)
    mask = np.zeros(len(indexKD), dtype=bool)
    for nIds in ballIter:
        nId = nIds[np.argmin(attributes[nIds])]
        mask[nId] = True
    return mask.nonzero()


def has_neighbour(indexKD, r):
    """Decide whether or not points have neighbouring points.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index of `n` points to filter.
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
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
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

# TODO: difference to has_neighbour? (only reverse?)


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
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
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


def ball(indexKD, r, order=None, inverse=False, axis=-1, min_pts=1):
    """Filter coordinates by radius.

    Parameters
    ----------
    indexKD : IndexKD
        IndexKD containing `n` points to filter.
    r : positive float or array_like(float, shape=(n))
        Ball radius or radii to apply.
    order : array_like(int, shape=(m))
        Order to proceed. If m < n, only a subset of points is investigated.
    axis : optional, int
        Axis to use for generating the order.
    min_pts : optional int
        Specifies how many neighbouring points within radius `r` are needed to
        yield a filtered point.

    Yields
    ------
    positive int
        Indices of the filtered points.

    Notes
    -----
    The filter guarantees the distance of neighboured points in a range of
    ]r, 2*r[.

    """

    # validation
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be an instance of 'IndexKD'")
    coords = indexKD.coords

    if order is None:
        order = np.argsort(coords[:, axis])[::-1]
    order = assertion.ensure_indices(order, max_value=len(indexKD) - 1)
    if inverse:
        order = order[::-1]

    if not hasattr(r, '__len__'):
        r = np.repeat(r, len(indexKD))
    r = assertion.ensure_numvector(r)
    if not np.all(r) > 0:
        raise ValueError("radius greater zero required")

    # filtering
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


def dem_filter(coords, r):
    """Select points suitable for generating a digital elevation model.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions to filter.
    r : Number or array_like(Number, shape=(n))
        Radius or radii to apply..

    Returns
    -------
    array_like(int, shape=(m))
        Desired indices of points suitable for generating a surface model.

    See Also
    --------
    radial_dem_filter

    """
    coords = assertion.ensure_coords(coords, min_dim=3)
    order = np.argsort(coords[:, -1])

    if not hasattr(r, '__getitem__') and assertion.isnumeric(r):
        r = np.repeat(r, len(order))
    r = assertion.ensure_numvector(r)

    # filter
    ball_gen = ball(IndexKD(coords[:, :-1]), r, order=order)
    fIds = np.array(list(ball_gen), dtype=int)

    # ensure neighbours
    count = IndexKD(coords[fIds, :]).ball_count(2 * r[fIds])
    fIds = fIds[count >= 6]

    # subsequent filter
    while True:
        m = len(fIds)
        count = IndexKD(coords[fIds, :]).ball_count(2 * r[fIds])
        fIds = fIds[count >= 3]
        if m == len(fIds):
            break

    return fIds


def radial_dem_filter(coords, angle_res, center=None):
    """Filter surface points with decreasing resolution with increasing
    distance to coordinate center.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Points to filter.
    angle_res : positive Number
        Filter resolution expressed as an angle.
    center : optional, array_like(Number, shape=(k))
        Desired center.

    Returns
    -------
    array_like(int, shape=(m))
        Indices of filtered points.

    See Also
    --------
    dem_filter

    """
    coords = assertion.ensure_coords(coords, min_dim=3)
    if center is None:
        center = np.zeros(coords.shape[1], dtype=float)
    center = assertion.ensure_numvector(center, min_length=3)
    if not (assertion.isnumeric(angle_res) and angle_res > 0):
        raise ValueError('angle greater zero required')

    dist = distance.dist(center[:-1], coords[:, :-1])
    radii = dist * np.sin(angle_res / 180.0 * np.pi)
    fIds = dem_filter(coords, radii)
    return fIds
