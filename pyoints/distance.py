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
"""Various distance metrics.
"""

import numpy as np

from . import (
    assertion,
)
from .misc import print_rounded


def norm(coords):
    """Normalization of coordinates.

    Parameters
    ----------
    coords : array_like(Number, shape=(k)) or array_like(Number, shape=(n, k))
        Represents `n` points or a single point of `k` dimensions.

    Returns
    -------
    array_like(shape=(n, ))
        Normed values.

    See Also
    --------
    snorm

    Examples
    --------

    >>> coords = [(3, 4), (0, 1), (4, 3), (0, 0), (8, 6)]
    >>> print_rounded(norm(coords))
    [  5.   1.   5.   0.  10.]

    """
    return np.sqrt(snorm(coords))


def snorm(coords):
    """Squared normalization of coordinates.

    Parameters
    ----------
    coords: array_like(Number, shape=(k)) or array_like(Number, shape=(n, k))
        Represents `n` points or a single point of `k` dimensions.

    Returns
    -------
    Number or array_like(Number, shape=(n))
        Squared normed values.

    See Also
    --------
    norm

    Examples
    --------

    >>> coords = [(3, 4), (0, 1), (4, 3), (0, 0), (8, 6)]
    >>> print_rounded(snorm(coords))
    [ 25   1  25   0 100]

    """
    coords = assertion.ensure_numarray(coords)
    if len(coords.shape) == 1:
        res = (coords * coords).sum()
    else:
        res = (coords * coords).sum(1)
    return res


def dist(p, coords):
    """Calculates the distances between points.

    Parameters
    ----------
    p : array_like(Number, shape=(n, k)) or array_like(Number, shape=(k))
        Represents `n` points or a single point of `k` dimensions.
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.

    Returns
    -------
    Number or array_like(Number, shape=(n))
        Normed values.

    See Also
    --------
    sdist

    Examples
    --------

    Point to points distance.

    >>> p = (1, 2)
    >>> coords = [(2, 2), (1, 1), (1, 2), (9, 8)]
    >>> print_rounded(dist(p, coords))
    [  1.   1.   0.  10.]

    Points to points distance.

    >>> A = [(2, 2), (1, 1), (1, 2)]
    >>> B = [(4, 2), (2, 1), (9, 8)]
    >>> print_rounded(dist(A, B))
    [  2.   1.  10.]

    """
    return np.sqrt(sdist(p, coords))


def sdist(p, coords):
    """Calculates the squared distances between points.

    Parameters
    ----------
    p : array_like(Number, shape=(n, k)) or array_like(Number, shape=(k))
        Represents `n` points or a single point of `k` dimensions.
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.

    Returns
    -------
    Number or array_like(Number, shape=(n))
        Squared distances between the points.

    See Also
    --------
    dist

    Examples
    --------

    Squared point to points distance.

    >>> p = (1, 2)
    >>> coords = [(2, 4), (1, 1), (1, 2), (9, 8)]
    >>> print_rounded(sdist(p, coords))
    [  5   1   0 100]

    Squared points to points distance.

    >>> A = [(2, 2), (1, 1), (1, 2)]
    >>> B = [(4, 2), (2, 1), (9, 8)]
    >>> print_rounded(sdist(A, B))
    [  4   1 100]

    """
    p = assertion.ensure_numarray(p)
    coords = assertion.ensure_numarray(coords)
    if not p.shape == coords.shape:
        if not (len(coords.shape) == 2 and p.shape[0] == coords.shape[1]):
            m = "Dimensions %s and %s do not match"
            raise ValueError(m % (str(p.shape), str(coords.shape)))

    return snorm(coords - p)


def rmse(A, B=None):
    """Calculates the Root Mean Squared Error of corresponding data sets.

    Parameters
    ----------
    A, B : array_like(Number, shape=(n, k))
        Represent `n` points or a single point of `k` dimensions.

    Returns
    -------
    Number
        Root Mean Squared Error.


    Examples
    --------

    >>> A = [(2, 2), (1, 1), (1, 2)]
    >>> B = [(2.2, 2), (0.9, 1.1), (1, 2.1)]
    >>> print_rounded(rmse(A, B))
    0.15

    """
    if B is None:
        d = snorm(A)
    else:
        d = sdist(A, B)
    return np.sqrt(np.mean(d))


def idw(dists, p=2):
    """Calculates the weights for Inverse Distance Weighting method.

    Parameters
    ----------
    dists : Number or array_like(Number, shape=(n))
        Represent `n` distance values.
    p : optional, Number
        Weighting power.

    Returns
    -------
    Number or array_like(Number, shape=(n))
        Weights according to Inverse Distance Weighting.

    Examples
    --------

    >>> dists = [0, 1, 4]

    >>> print_rounded(idw(dists))
    [ 1.    0.25  0.04]

    >>> print_rounded(idw(dists, p=1))
    [ 1.   0.5  0.2]

    """
    dists = assertion.ensure_numvector(dists)
    return 1.0 / (1 + dists)**p
