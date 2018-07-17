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


def _preferred_normals(normals, shape):
    # helper function
    if normals is None:
        normals = np.zeros(shape[1])
        normals[-1] = 1
    normals = assertion.ensure_numarray(normals)
    if len(normals.shape) == 1:
        normals = np.tile(normals, (shape[0], 1))
    normals = assertion.ensure_coords(normals, dim=shape[1])
    normals = (normals.T / distance.norm(normals)).T
    return normals


def fit_normals(coords, radii, indices=None, preferred_normals=None):
    """Fit normals to points points.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.
    radii : positive Number or array_like(Number, shape=(n))
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

    # define prefered normals
    preferred_normals = _preferred_normals(preferred_normals, coords.shape)

    indexKD = coords.indexKD()
    if assertion.isnumeric(radii):
        ball_gen = indexKD.ball_iter(coords[indices, :], radii)
    else:
        radii = assertion.ensure_numvector(radii, length=len(indices))
        ball_gen = indexKD.balls_iter(coords[indices, :], radii)

    # generate normals
    normals = np.zeros((len(indices), dim), dtype=float)
    for pId, nIds in enumerate(ball_gen):
        if len(nIds) >= dim:
            normals[pId, :] = PCA(coords[nIds, :]).pc(dim)

    # flip normals if required
    dists = distance.snorm(normals - preferred_normals)
    normals[dists > 2] *= -1

    return normals


def approximate_normals(coords, radii, preferred_normals=None):
    """Calculate approximate normals of points.

    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.
    radii : positive Number or array_like(Number, shape=(n))
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
    if assertion.isnumeric(radii):
        radii = np.repeat(radii, len(coords))
    else:
        radii = assertion.ensure_numvector(radii, length=len(coords))

    # define prefered normals
    preferred_normals = _preferred_normals(preferred_normals, coords.shape)

    # generate normals
    normals = np.zeros(coords.shape, dtype=float)
    for pId in range(len(coords)):
        if normals[pId].sum() == 0:
            nIds = indexKD.ball(coords[pId, :], radii[pId])
            if len(nIds) >= dim:
                normals[nIds, :] = PCA(coords[nIds, :]).pc(dim)

    # flip normals if required
    dists = distance.snorm(normals - preferred_normals)
    normals[dists > 2] *= -1

    return normals
