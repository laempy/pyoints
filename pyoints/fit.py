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
"""Fits shapes or functions to points.
"""

import numpy as np
import cylinder_fitting

from . import (
    transformation,
    assertion,
    vector,
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

    Draw points on a half circle with radius 5 and center (2, 4) and try to
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
    """Fits a cylinder to points.

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
