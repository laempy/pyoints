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
"""Fit shapes or functions to points.
"""

import numpy as np
import cylinder_fitting

from . import (
    transformation,
    assertion,
    vector,
    distance,
    Coords,
    IndexKD,
    filters,
)


def fit_sphere(coords, weights=1.0):
    """Least square fitting of a sphere to a set of points.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents n data points of `k` dimensions.
    weights : (k+1,k+1), array_like
        Transformation matrix.

    Returns
    -------
    center : np.ndarray(Number, shape=(k))
        Center of the sphere.
    r : Number
        Radius of the sphere.

    Notes
    -----
    Idea taken from [1].

    References
    ----------

    [1] A. Bruenner (2001): URL
    http://www.arndt-bruenner.de/mathe/scripts/kreis3p.htm

    Examples
    --------

    Draw points on a half circle with radius 5 and cener (2, 4) and try to
    dertermine the circle parameters.

    >>> x = np.arange(-1, 1, 0.1)
    >>> y = np.sqrt(5**2 - x**2)
    >>> coords = np.array([x,y]).T + [2,4]
    >>> center, r, residuals = fit_sphere(coords)
    >>> print(center)
    [2. 4.]
    >>> print(np.round(r, 2))
    5.0

    """
    coords = assertion.ensure_coords(coords)
    dim = coords.shape[1]

    if not assertion.isnumeric(weights):
        weights = assertion.ensure_numvector(weights, length=dim)

    # mean-centering to avoid overflow errors
    c = coords.mean(0)
    cCoords = coords - c

    # create matrices
    A = transformation.homogenious(cCoords, value=1)
    B = (cCoords**2).sum(1)

    A = (A.T * weights).T
    B = B * weights

    # solve equation system
    p, residuals, rank, s = np.linalg.lstsq(A, B, rcond=-1)

    bCenter = 0.5 * p[:dim]
    r = np.sqrt((bCenter**2).sum() + p[dim])
    center = bCenter + c

    return center, r, residuals


def fit_cylinder(coords, vec=None):
    """Fit a cylinder to points.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents n data points of `k` dimensions.
    vec : optional, array_like(Number, shape(k))
        Estimated orientation of the cylinder axis.

    Returns
    -------
    vec: vector.Vector
        Orientaton vector.
    r : Number
        Radius of the cylinder.
    resid : Number
        Remaining residuals.

    Examples
    --------

    Prepare roto-translated half cylinder.

    >>> r = 2.5
    >>> x = np.arange(-1, 0, 0.01) * r
    >>> y = np.sqrt(r**2 - x**2)
    >>> y[::2] = - y[::2]
    >>> z = np.repeat(5, len(x))
    >>> z[::2] = -5
    >>> T = transformation.matrix(t=[10, 20, 30], r=[0.3, 0.2, 0.0])
    >>> coords = transformation.transform(np.array([x, y, z]).T, T)

    Get cylinder.

    >>> vec, r, residuals = fit_cylinder(coords, vec=[0, 0, 1])

    >>> print(np.round(r, 2))
    2.5
    >>> print(np.round(vec.origin, 2))
    [10. 20. 30.]

    Check distances to vector.

    >>> dists = vec.distance(coords)
    >>> print(np.round([np.min(dists), np.max(dists)], 2))
    [2.5 2.5]

    """
    coords = assertion.ensure_coords(coords, dim=3)

    # set estimated direction
    if vec is not None:
        vec = assertion.ensure_numvector(vec, length=3)
        phi, theta = vector.direction(vec)
        guess_angles = [(phi, theta)]
    else:
        guess_angles = None

    # fit cylinder
    vec, origin, r, residuals = cylinder_fitting.fit(
        coords,
        guess_angles=guess_angles
    )
    v = vector.Vector(origin, vec)

    return v, r, residuals


def _orient_normals(normals, p_normals):
    # helper function

    shape = normals.shape

    if p_normals is None:
        p_normals = np.zeros(shape[1])
        p_normals[-1] = 1
    p_normals = assertion.ensure_numarray(p_normals)
    if len(p_normals.shape) == 1:
        p_normals = np.tile(p_normals, (shape[0], 1))
    p_normals = assertion.ensure_coords(p_normals, dim=shape[1])
    p_normals = (p_normals.T / distance.norm(p_normals)).T

    # orient normals
    dist = distance.sdist(normals, p_normals)
    normals[dist > shape[1], :] *= -1

    return normals


def fit_normals_ball(coords, r, indices=None, preferred_normals=None):
    """Fits normals to points, by selecting all neighbours within a sphere.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.
    r : positive Number or array_like(Number, shape=(n))
        Radius or radii to select neighbouring points.
    indices : optional, array_like(int, shape=(m))
        Vector of point indices to subsample the point cloud (`m <= n`). If
        None, `indices` is set to `range(n)`.
    preferred_normals : optional, array_like(Number, shape=(m, k))
        Preferred orientation of the normals.

    Returns
    -------
    array_like(Number, shape=(m, k))
        Desired normals of coordinates `coords`.

    Examples
    --------

    Two dimensional normals.

    >>> coords = [(0, 0), (1, 1), (2, 3), (3, 3), (4, 2), (5, 1), (5, 0)]
    >>> normals = fit_normals_ball(coords, 1.5)
    >>> print(np.round(normals, 2))
    [[-0.71  0.71]
     [-0.71  0.71]
     [ 0.    1.  ]
     [ 0.47  0.88]
     [ 0.71  0.71]
     [ 0.88  0.47]
     [ 1.    0.  ]]

    """
    coords = Coords(coords)
    dim = coords.dim

    # subset
    if indices is None:
        indices = np.arange(len(coords))
    else:
        indices = assertion.ensure_numvector(indices, max_length=len(coords))

    indexKD = coords.indexKD()

    if assertion.isnumeric(r):
        ball_gen = indexKD.ball_iter(coords[indices, :], r)
    else:
        r = assertion.ensure_numvector(r, length=len(indices))
        ball_gen = indexKD.balls_iter(coords[indices, :], r)

    # generate normals
    normals = np.zeros((len(indices), dim), dtype=float)
    for pId, nIds in enumerate(ball_gen):
        if len(nIds) >= dim:
            eig_vec = transformation.eigen(coords[nIds, :])[0][:, -1]
            normals[pId, :] = eig_vec

    # flip normals if required
    normals = _orient_normals(normals, preferred_normals)

    return normals


def fit_normals_knn(
        coords,
        k=3,
        distance_upper_bound=np.inf,
        indices=None,
        preferred_normals=None):
    """Fits normals to points by selecting `k` nearest neighbours.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.
    k : positive int
        Specifies the number of neighbours to select.
    distance_upper_bound : optional, positive Number
        Maximum radius to select neighbouring points.
    indices : optional, array_like(int, shape=(m))
        Vector of point indices to subsample the point cloud (`m <= n`). If
        None, `indices` is set to `range(n)`.
    preferred_normals : optional, array_like(Number, shape=(m, k))
        Preferred orientation of the normals.

    Returns
    -------
    array_like(Number, shape=(m, k))
        Desired normals of coordinates `coords`.

    Examples
    --------

    Two dimensional normals.

    >>> coords = [(0, 0), (1, 1), (2, 3), (3, 3), (4, 2), (5, 1), (5, 0)]
    >>> normals = fit_normals_knn(coords, 2)
    >>> print(np.round(normals, 2))
    [[-0.71  0.71]
     [-0.71  0.71]
     [ 0.    1.  ]
     [ 0.    1.  ]
     [ 0.71  0.71]
     [ 1.    0.  ]
     [ 1.    0.  ]]

    """
    coords = Coords(coords)
    dim = coords.dim

    # subset
    if indices is None:
        indices = np.arange(len(coords))
    else:
        indices = assertion.ensure_numvector(indices, max_length=len(coords))

    indexKD = coords.indexKD()

    if not (isinstance(k, int) and k > 0):
        raise ValueError("'k' needs to be an integer greater zero")
    if not (assertion.isnumeric(distance_upper_bound) and
            distance_upper_bound > 0):
        m = "'distance_upper_bound' needs to be an number greater zero"
        raise ValueError(m)

    knn_gen = indexKD.knn_iter(
        coords[indices, :], k=k, distance_upper_bound=distance_upper_bound)

    # generate normals
    normals = np.zeros((len(indices), dim), dtype=float)
    for pId, (dists, nIds) in enumerate(knn_gen):
        nIds = [nId for dist, nId in zip(dists.flatten(), nIds.flatten())
                if dist < distance_upper_bound]
        if len(nIds) >= dim:
            eig_vec = transformation.eigen(coords[nIds, :])[0][:, -1]
            normals[pId, :] = eig_vec

    # flip normals if required
    normals = _orient_normals(normals, preferred_normals)

    return normals


def approximate_normals(coords, r, preferred_normals=None):
    """Calculate approximate normals of points.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.
    r : positive Number or array_like(Number, shape=(n))
        Radius or radii to select neighbouring points.
    preferred_normals : array_like(Number, shape=(n, k))
        Preferred orientation of the normals.

    Returns
    -------
    array_like(Number, shape=(n, k))
        Desired normals of coordinates `coords`.

    Examples
    --------

    Two dimensional normals.

    >>> coords = [(0, 0), (1, 1), (2, 3), (3, 3), (4, 2), (5, 1), (5, 0)]
    >>> normals = approximate_normals(coords, 1.5)
    >>> print(np.round(normals, 2))
    [[-0.71  0.71]
     [-0.71  0.71]
     [ 0.    1.  ]
     [ 0.71  0.71]
     [ 0.71  0.71]
     [ 1.    0.  ]
     [ 1.    0.  ]]

    """
    coords = Coords(coords)
    indexKD = coords.indexKD()
    dim = coords.dim

    # check radii
    if assertion.isnumeric(r):
        r = np.repeat(r, len(coords))
    else:
        r = assertion.ensure_numvector(r, length=len(coords))

    # generate normals
    if preferred_normals is None:
        normals = np.zeros(coords.shape, dtype=float)
    else:
        preferred_normals = assertion.ensure_coords(preferred_normals, dim=dim)
        if not preferred_normals.shape == coords.shape:
            raise ValueError("unexpected shape of 'preferred_normals'")
        length = distance.norm(preferred_normals)
        preferred_normals = (preferred_normals.T / length.T).T
        normals = assertion.ensure_coords(preferred_normals, dim=dim)

    not_visited = np.ones(len(coords), dtype=bool)
    for pId in range(len(coords)):
        if not_visited[pId]:
            nIds = indexKD.ball(coords[pId, :], r[pId])
            if len(nIds) >= dim:
                eig_vec = transformation.eigen(coords[nIds, :])[0][:, -1]
                normals[nIds, :] = eig_vec
                not_visited[nIds] = False

    normals = _orient_normals(normals, preferred_normals)

    return normals
