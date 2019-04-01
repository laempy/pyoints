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
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE
"""Point filters.
"""

import numpy as np
from numbers import Number

from . import (
    assertion,
    distance,
    interpolate,
    IndexKD,
    transformation,
    vector,
)

from .misc import print_rounded


def extrema(indexKD, attributes, r, inverse=False):
    """Finds local maxima or minima of given point set.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index of `n` points to filter.
    attributes : array_like(Number, shape=(n))
        Attributes to search for extrema. If none, the last coordinate
        dimension is used as the attribute.
    r : positive Number
        Maximum distance between two points to be considered as neighbours.
    inverse : optional, bool
        Indicates if local maxima (False) or local minima (True) shall be
        yielded.

    Yields
    ------
    positive int
        Indices of local maxima or minima.

    Examples
    --------

    Find local maxima.

    >>> indexKD = IndexKD([(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0.5) ])
    >>> attributes = [2, 0.1, 1, 0, 0.5]
    >>> fIds = list(extrema(indexKD, attributes, 1.1))
    >>> print_rounded(fIds)
    [0 2]

    Find local minima.

    >>> fIds = list(extrema(indexKD, attributes, 1.1, inverse=True))
    >>> print_rounded(fIds)
    [3 1]

    >>> fIds = list(extrema(indexKD, attributes, 1.5, inverse=True))
    >>> print_rounded(fIds)
    [3]

    """
    # type validation
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be an instance of 'IndexKD'")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs to be a number greater zero")
    attributes = assertion.ensure_numvector(attributes, length=len(indexKD))
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
    """Finds minima or maxima within a specified radius for all points. For
    each point the neighbouring point with extreme attribute is marked.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index of `n` points to filter.
    attributes : array_like(Number, shape=(n))
        Attributes to search for extrema. If none, the last coordinate
        dimension is used as the attribute.
    r : positive Number
        Local radius to search for a minimum.
    inverse : optional, bool
        Indicates if minima (False) or maxima (True) shall be identfied.

    Returns
    -------
    np.ndarray(int, shape=(n))
        Filtered point indices.

    """
    # type validation
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs to be a number greater zero")

    attributes = assertion.ensure_numvector(attributes, length=len(indexKD))
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
    """Filters points which have neighbours within a specific radius.

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
        Index of a point with at least one neighbouring point with radius `r`.

    See also
    --------
    is_isolated

    Examples
    --------

    >>> coords = [(0, 0), (0.5, 0.5), (0, 1), (0.7, 0.5), (-1, -1)]
    >>> indexKD = IndexKD(coords)
    >>> print_rounded(list(has_neighbour(indexKD, 0.7)))
    [1 3]

    """
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs to be a number greater zero")

    not_classified = np.ones(len(indexKD), dtype=np.bool)
    for pId, coord in enumerate(indexKD.coords):
        if not_classified[pId]:
            nIds = indexKD.ball(coord, r)
            if len(nIds) > 1:
                not_classified[nIds] = False
                yield pId
        else:
            yield pId


def is_isolated(indexKD, r):
    """Filters points which have no neighbours within a specific radius.

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
        Index of an isolated point with no neighbouring point within radius
        `r`.

    See Also
    --------
    has_neighbour

    Examples
    --------

    >>> coords = [(0, 0), (0.5, 0.5), (0, 1), (0.7, 0.5), (-1, -1)]
    >>> indexKD = IndexKD(coords)
    >>> print_rounded(list(is_isolated(indexKD, 0.7)))
    [0 2 4]

    """
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs to be a number greater zero")

    not_classified = np.ones(len(indexKD), dtype=np.bool)
    for pId, coord in enumerate(indexKD.coords):
        if not_classified[pId]:
            nIds = indexKD.ball(coord, r)
            if len(nIds) > 1:
                not_classified[nIds] = False
            else:
                yield pId


def ball(indexKD, r, order=None, inverse=False, axis=-1, min_pts=1):
    """Filters coordinates by radius. This algorithm is suitable to remove
    duplicate points or to get an almost uniform point density.

    Parameters
    ----------
    indexKD : IndexKD
        IndexKD containing `n` points to filter.
    r : positive float or array_like(float, shape=(n))
        Ball radius or radii to select neighbouring points.
    order : optional, array_like(int, shape=(m))
        Order to proceed. If m < n, only a subset of points is investigated. If
        none, ordered by `axis`.
    axis : optional, int
        Coordinate axis to use to generate the order.
    inverse : bool
        Indicates whether or not the `order` is inversed.
    min_pts : optional, int
        Specifies how many neighbouring points within radius `r` shall be
        required to yield a filtered point. This parameter can be used to
        filter noisy point sets.

    Yields
    ------
    positive int
        Indices of the filtered points.

    Notes
    -----
    Within a dense point cloud, the filter guarantees the distance of
    neighboured points in a range of `]r, 2*r[`.

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
    not_classified = np.ones(len(indexKD), dtype=np.bool)
    for pId in order:
        if not_classified[pId]:
            nIds = indexKD.ball(coords[pId, :], r[pId])
            if len(nIds) >= min_pts:
                not_classified[nIds] = False
                yield pId


def in_convex_hull(hull_coords, coords):
    """Decides whether or not points are within a convex hull.

    Parameters
    ----------
    hull_coords : array_like(Number, shape=(m, k))
        Represents `m` hull points of `k` dimensions.
    coords : array_like(Number, shape=(n, k))
        Represents `n` data points to check whether or not they are located
        within the convex hull.

    Returns
    -------
    array_like(bool, shape=(n))
        Indicates whether or not the points are located within the convex hull.

    Examples
    --------

    >>> hull = [(0, 0), (1, 0), (1, 2)]
    >>> coords = [(0, 0), (0.5, 0.5), (0, 1), (0.7, 0.5), (-1, -1)]
    >>> print(in_convex_hull(hull, coords))
    [ True  True False  True False]

    """
    hull_coords = assertion.ensure_coords(hull_coords)
    coords = assertion.ensure_coords(coords)

    interpolator = interpolate.LinearInterpolator(
        hull_coords,
        np.zeros(hull_coords.shape[0])
    )
    return ~np.isnan(interpolator(coords))


def surface(indexKD, r, order=None, inverse=False, axis=-1):
    """Filters points associated with a surface.

    Parameters
    ----------
    indexKD : IndexKD
        IndexKD containing `n` points to filter.
    r : positive float or array_like(float, shape=(n))
        Ball radius or radii to apply.
    order : optional, array_like(int, shape=(m))
        Order to proceed. If m < n, only a subset of points is investigated.
    inverse : optional, bool
        Indicates whether or not to inverse the order.
    axis : optional, int
        Axis to use for generating the order.

    Yields
    ------
    positive int
        Indices of the filtered points.

    """
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs to be a number greater zero")

    coords = indexKD.coords

    if order is None:
        order = np.argsort(coords[:, axis])[::-1]
    else:
        order = assertion.ensure_indices(order, max_value=len(coords) - 1)

    if inverse:
        order = order[::-1]
    inverseOrder = np.argsort(order)

    not_classified = np.zeros(len(indexKD), dtype=np.bool)
    not_classified[order] = True
    for pId in order:
        if not_classified[pId]:
            nIds = np.array(indexKD.ball(coords[pId, :], r))
            not_classified[nIds] = False
            yield nIds[np.argmin(inverseOrder[nIds])]


def dem_filter(coords, r, max_angle=70):
    """Selects points suitable for generating a digital elevation model.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions to filter.
    r : Number or array_like(Number, shape=(n))
        Radius or radii to apply.
    max_angle : Number
        Maximum allowed slope of a simplex.

    Returns
    -------
    array_like(int, shape=(m))
        Desired indices of points suitable for generating a surface model.

    Notes
    -----
    Applies a `k-1` dimensional `ball` filter to identify a digital elevation
    model. The point order is defined by the last coordinate dimension. The
    filtering radius `r` is defined by the user. To optimize the quality of the
    dem, all points with less than 5 neighbours within radius `2 * r` are
    removed from further analysis. In addition a TIN (Triangulated Irregular
    Network) is generated to model the surface. Each facet with a slope larger
    than `max_angle` is removed iteratively.

    See Also
    --------
    radial_dem_filter, ball

    """
    coords = assertion.ensure_coords(coords, min_dim=3)
    order = np.argsort(coords[:, -1])

    if not hasattr(r, '__getitem__'):
        r = np.repeat(r, len(order))
    r = assertion.ensure_numvector(r)

    if not (isinstance(max_angle, Number) and max_angle > 0):
        raise ValueError("'max_angle' needs to be a number greater zero")

    # filter
    ball_gen = ball(IndexKD(coords[:, :-1]), r, order=order)
    fIds = np.array(list(ball_gen), dtype=int)

    # ensure neighbours
    count = IndexKD(coords[fIds, :]).ball_count(2 * r[fIds])
    fIds = fIds[count >= 6]

    # subsequent filtering of the simplices
    while True:
        old_len = len(fIds)

        fcoords = coords[fIds, :]
        dem = interpolate.LinearInterpolator(fcoords[:, :-1], fcoords[:, -1])

        # check principal component of the simplices
        mask = np.ones(len(fIds), dtype=bool)
        for simplex_indices in dem.delaunay.simplices:
            simplex = fcoords[simplex_indices, :]
            eig_vec = transformation.eigen(simplex)[0][:, -1]
            angle = vector.zenith(eig_vec, deg=True)
            if angle > max_angle:
                idx = simplex_indices[np.argmax(simplex[:, -1])]
                mask[idx] = False
        fIds = fIds[mask]

        if old_len == len(fIds):
            break

    return fIds


def radial_dem_filter(coords, angle_res, center=None, max_angle=70):
    """Filters surface points based on distance to the center. The algorithm
    is designed to create digital elevation models of terrestrial laser scans.
    Terrestrial laser scans are characterized by decreasing point densities
    with increasing distance to the scanner position. Thus, the radius to
    identify neighbouring points is adjusted to the distance of the scanner.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Points to filter.
    angle_res : positive Number
        Filter resolution expressed as an angle.
    center : optional, array_like(Number, shape=(k))
        Desired center.
    max_angle : Number
        Maximum allowed angle of a simplex.

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
    center = assertion.ensure_numvector(center, length=3)

    if not (assertion.isnumeric(angle_res) and angle_res > 0):
        raise ValueError('angle greater zero required')

    dist = distance.dist(center[:-1], coords[:, :-1])
    radii = dist * np.sin(angle_res / 180.0 * np.pi)
    fIds = dem_filter(coords, radii, max_angle=max_angle)
    fIds = fIds[coords[fIds, -1] < center[-1]]
    return fIds
