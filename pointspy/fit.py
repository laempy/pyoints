"""Fit shapes or functions to points.
"""

import numpy as np
import cylinder_fitting
from scipy import optimize


from . import (
    nptools,
    transformation,
    assertion,
    distance,
    vector,
)


def sphere(coords, weights=1.0):
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
    >>> center, r, residuals = ball(coords)
    >>> print center
    [2. 4.]
    >>> print np.round(r, 2)
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


def cylinder(coords, vec=None):
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

    >>> vec, r, residuals = cylinder(coords, vec=[0, 0, 1])

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
