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
"""Derive normals of point clouds.
"""

import numpy as np
from numbers import Number

from . import (
    Coords,
    assertion,
    distance,
)
from .transformation import eigen


def prefer_orientation(normals, preferred):
    """Orients normals using a prefered normals.

    Parameters
    ----------
    normals : array_like(Number, shape=(n, k))
        Normals of `n` points with `k` dimensions.
    preferred : array_like(Number, shape=(k)) or array_like(Number, shape=(n, k))
        Preferred normal orientation for each normal in `normals`.

    Returns
    -------
    np.ndarray(Number, shape=(n, k))
        Oriented normals. If the angle between a normal in `normals` and the
        corresponding normal in `preferred` is greater than 90 degree, the
        it is flipped.

    """
    normals = assertion.ensure_coords(normals)
    preferred = normalize(preferred)

    if len(preferred.shape) == 1:
        preferred = np.tile(preferred, (normals.shape[0], 1))

    # replace missing normals
    mask = np.all(normals == 0, axis=1)
    normals[mask, :] = preferred[mask, :]

    # orient normals
    sdist = distance.sdist(normals, preferred)

    normals[sdist > normals.shape[1], :] *= -1

    return normals


def normalize(vectors, dim=None, n=None):
    """Make a vector or vectors a set of normals with a given shape.

    Parameters
    ----------
    vectors : array_like(Number, shape=(k)) or array_like(Number, shape=(n, k))
        Orientation of the normals.
    shape : (n, k)
        Desired shape of the output normals.

    Returns
    -------
    np.ndarray(Number, shape=(n, k))
        Normals with ensured properties.

    Examples
    --------

    Normalize two dimensional vectors.

    >>> vectors = [(3, 4), (8, 6), (2, 0), (0, 0)]
    >>> normals = normalize(vectors)

    >>> print(normals)
    [[0.6 0.8]
     [0.8 0.6]
     [1.  0. ]
     [0.  0. ]]
    >>> print(distance.norm(normals))
    [1. 1. 1. 0.]

    Normalize three dimensional vectors.

    >>> vectors = [(3, 0, 4), (2, 0, 0), (0, 0, 0)]
    >>> normals = normalize(vectors)

    >>> print(normals)
    [[0.6 0.  0.8]
     [1.  0.  0. ]
     [0.  0.  0. ]]
    >>> print(distance.norm(normals))
    [1. 1. 0.]

    Normalize individual vectors.

    >>> print(normalize([3, 4]))
    [0.6 0.8]
    >>> print(normalize((3, 0, 4)))
    [0.6 0.  0.8]

    """
    vectors = assertion.ensure_numarray(vectors)
    if not len(vectors.shape) in (1, 2):
        raise ValueError("'vectors' need be a one or two dimensional array")

    lengths = distance.norm(vectors)
    if len(vectors.shape) == 1:
        lengths = 1 if lengths == 0 else lengths
    else:
        lengths[lengths == 0] = 1
    return (vectors.T / lengths).T


def fit_normals(
        coords,
        r=np.inf,
        k=None,
        indices=None,
        preferred=None):
    """Fits normals to points by selecting nearest neighbours.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, dim))
        Represents `n` points of `dim` dimensions.
    r : optional, positive Number
        Maximum radius to select neighbouring points.
    k : optional, positive int
        Specifies the number of neighbours to select. If specified, at least
        `dim` neighbours are required.
    indices : optional, array_like(int, shape=(m))
        Vector of point indices to subsample the point cloud (`m <= n`). If
        None, `indices` is set to `range(n)`.
    preferred : optional, array_like(Number, shape=(k)) or array_like(Number, shape=(n, k))
        Preferred orientation of the normals.

    Returns
    -------
    array_like(Number, shape=(m, k))
        Desired normals of coordinates `coords`.

    See Also
    --------
    approximate_normals, prefer_orientation

    Examples
    --------

    >>> coords = [(0, 0), (1, 1), (2, 3), (3, 3), (4, 2), (5, 1), (5, 0)]

    Fit normals using `k` nearest neighbours.

    >>> normals = fit_normals(coords, k=2, preferred=[1, 0])
    >>> print(np.round(normals, 2))
    [[ 0.71 -0.71]
     [ 0.71 -0.71]
     [ 0.    1.  ]
     [ 0.    1.  ]
     [ 0.71  0.71]
     [ 1.    0.  ]
     [ 1.    0.  ]]

    Fit normals a using all nearest neighbours within radius `r`.

    >>> normals = fit_normals(coords, r=2.5, preferred=[1, 0])
    >>> print(np.round(normals, 2))
    [[ 0.71 -0.71]
     [ 0.84 -0.54]
     [ 0.45 -0.89]
     [ 0.47  0.88]
     [ 0.71  0.71]
     [ 0.88  0.47]
     [ 0.88  0.47]]

    Fit normals using `k` nearest neighbours within radius `r`.

    >>> normals = fit_normals(coords, r=2.5, k=3, preferred=[1, 0])
    >>> print(np.round(normals, 2))
    [[ 0.71 -0.71]
     [ 0.84 -0.54]
     [ 0.76 -0.65]
     [ 0.47  0.88]
     [ 0.71  0.71]
     [ 0.88  0.47]
     [ 0.88  0.47]]

    """
    coords = Coords(coords)
    indexKD = coords.indexKD()
    dim = coords.dim

    # subset
    if indices is None:
        indices = np.arange(len(coords))
    else:
        indices = assertion.ensure_numvector(indices, max_length=len(coords))

    if preferred is not None:
        preferred = normalize(preferred)

    if not (isinstance(r, Number) and r > 0):
        raise ValueError("'r' needs to be a Number greater zero")

    if k is None:
        nids_gen = indexKD.ball_iter(coords[indices, :], r)
    else:
        if not (isinstance(k, int) and k >= dim):
            m = "'k' needs to be an integer greater or equal %i" % dim
            raise ValueError(m)

        def gen():
            knn_gen = indexKD.knn_iter(
                coords[indices, :], k=k, distance_upper_bound=r)
            for (dists, nids) in knn_gen:
                yield nids[dists < r]
        nids_gen = gen()

    # generate normals
    normals = np.zeros((len(indices), dim), dtype=float)
    for pId, nIds in enumerate(nids_gen):
        if len(nIds) >= dim:
            eig_vec = eigen(coords[nIds, :])[0][:, -1]
            normals[pId, :] = eig_vec

    # flip normals if required
    if preferred is not None:
        normals = prefer_orientation(normals, preferred)

    return normals


def approximate_normals(coords, r=np.inf, k=None, preferred=None):
    """Approximates normals of points by selecting nearest neighbours and
    assigning the derived normal to all neighbours.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, dim))
        Represents `n` points of `dim` dimensions.
    r : optional, positive Number
        Maximum radius to select neighbouring points.
    k : optional, positive int
        Specifies the number of neighbours to select. If specified, at least
        `dim` neighbours are required.
    preferred : optional, array_like(Number, shape=(k)) or array_like(Number, shape=(n, k))
        Preferred orientation of the normals.

    Returns
    -------
    array_like(Number, shape=(n, dim))
        Approximated normals for the coordinates `coords`.

    See Also
    --------
    fit_normals, prefer_orientation

    Examples
    --------

    >>> coords = [(0, 0), (1, 1), (2, 3), (3, 3), (4, 2), (5, 1), (5, 0)]

    Approximate two normals using all neighbours within radius `n`.

    >>> normals = approximate_normals(coords, 2.5, preferred=(1, 0))
    >>> print(np.round(normals, 2))
    [[ 0.71 -0.71]
     [ 0.45 -0.89]
     [ 0.45 -0.89]
     [ 0.45 -0.89]
     [ 0.88  0.47]
     [ 0.88  0.47]
     [ 0.88  0.47]]

    Approximate two normals using `k` nearest neighbours.

    >>> normals = approximate_normals(coords, k=4, preferred=(1, 0))
    >>> print(np.round(normals, 2))
    [[ 0.76 -0.65]
     [ 0.76 -0.65]
     [ 0.59  0.81]
     [ 0.81  0.59]
     [ 0.81  0.59]
     [ 0.81  0.59]
     [ 0.81  0.59]]

    Approximate two normals using `k` nearest neighbours within radius `r`.

    >>> normals = approximate_normals(coords, k=4, r=2.5, preferred=(1, 0))
    >>> print(np.round(normals, 2))
    [[ 0.71 -0.71]
     [ 0.45 -0.89]
     [ 0.45 -0.89]
     [ 0.45 -0.89]
     [ 0.88  0.47]
     [ 0.88  0.47]
     [ 0.88  0.47]]

    """
    coords = Coords(coords)
    indexKD = coords.indexKD()
    dim = coords.dim

    if preferred is not None:
        preferred = normalize(preferred)

    if not (isinstance(r, Number) and r > 0):
        raise ValueError("'r' needs to be a Number greater zero")

    if k is None:
        def get_neigbours(coord):
            return indexKD.ball(coord, r)
    else:
        if not (isinstance(k, int) and k >= dim):
            m = "'k' needs to be an integer greater or equal %i" % dim
            raise ValueError(m)

        def get_neigbours(coord):
            dists, nids = indexKD.knn(coord, k=k, distance_upper_bound=r)
            return nids[dists < r]

    normals = np.zeros(coords.shape, dtype=float)
    not_visited = np.ones(len(coords), dtype=bool)
    for pId in range(len(coords)):
        if not_visited[pId]:
            nIds = get_neigbours(coords[pId, :])
            if len(nIds) >= dim:
                eig_vec = eigen(coords[nIds, :])[0][:, -1]
                normals[nIds, :] = eig_vec
                not_visited[nIds] = False

    # flip normals if required
    if preferred is not None:
        normals = prefer_orientation(normals, preferred)

    return normals
