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
    filters,
    IndexKD,
)
from .transformation import PCA

from .misc import *


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

    References
    ----------
    # http://www.arndt-bruenner.de/mathe/scripts/kreis3p.htm

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


def fit_normals(coords, r, indices=None, preferred_normals=None):
    """Fit normals to points points.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.
    r : positive Number or array_like(Number, shape=(n))
        Radius or radii to select neighbouring points.

    Returns
    -------
    array_like(Number, shape=(n, k))
        Desired normals of coordinates `coords`.

    Examples
    --------

    Two dimensional normals.

    >>> coords = [(0, 0), (1, 1), (2, 3), (3, 3), (4, 2), (5, 1), (5, 0)]
    >>> normals = fit_normals(coords, 1.5)
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
            eig_vec, eig_val = transformation.eigen(coords[nIds, :])
            normals[pId, :] = eig_vec[:, -1]

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
    normals = np.zeros(coords.shape, dtype=float)
    not_visited = np.ones(len(coords), dtype=bool)
    for pId in range(len(coords)):
        if not_visited[pId]:
            nIds = indexKD.ball(coords[pId, :], r[pId])
            if len(nIds) >= dim:
                eig_vec, eig_val = transformation.eigen(coords[nIds, :])
                normals[nIds, :] = eig_vec[:, -1]
                not_visited[nIds] = False

    normals = _orient_normals(normals, preferred_normals)

    return normals
