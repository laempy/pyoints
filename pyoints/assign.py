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
"""Module to find pairs of points"""

import numpy as np

from . import (
    assertion,
    transformation,
    IndexKD,
)

from .misc import print_rounded


class Matcher:
    """Base class to simplify point matching. Points of a reference point set
    `A` are assigned to points of a point set `B`.

    Parameters
    ----------
    A : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions. These points are used as a
        reference point set.
    radii : array_like(Number, shape=(k))
        Defines the sphere within the points can get assigned.

    """

    def __init__(self, coords, radii):
        coords = assertion.ensure_coords(coords)
        radii = assertion.ensure_numvector(radii, length=coords.shape[1])

        S = transformation.s_matrix(1.0 / radii)
        self.rIndexKD = IndexKD(coords, S)

    def __call__(coords):
        """Find matching points.

        Parameters
        ----------
        B : array_like(Number, shape=(n, k))
            Represents `n` points of `k` dimensions. These points are assigned
            to the previously defined reference coordinates.

        Returns
        -------
        pairs : np.ndarray(int, shape=(m, 2))
            Indices of assigned points. For two point sets `A`, `B` and each
            row `(a, b)` in `pairs` `A[a, :]` is assigned to `B[b, :]`

        """
        raise NotImplementedError()


class PairMatcher(Matcher):
    """Find unique pairs of points. A point `a` of point set `A` is assigned
    to its closest point `b` of point set `B` if `a` is also the nearest
    neighbour to `b`. Thus, duplicate assignment is not possible.

    See Also
    --------
    Matcher

    Examples
    --------

    Create point sets.

    >>> A = np.array([(0, 0), (0, 0.1), (1, 1), (1, 0), (0.5, 0.5), (-1, -2)])
    >>> B = np.array([(0.4, 0.4), (0.2, 0), (0.1, 1.2), (2, 1), (-1.1, -1.2)])

    Match points.

    >>> matcher = PairMatcher(A, [0.3, 0.2])
    >>> pairs = matcher(B)
    >>> print_rounded(pairs)
    [[4 0]
     [0 1]]
    >>> print_rounded(A[pairs[:, 0], :] - B[pairs[:, 1], :])
    [[ 0.1  0.1]
     [-0.2  0. ]]

    """

    def __init__(self, coords, radii):
        coords = assertion.ensure_coords(coords)
        radii = assertion.ensure_numvector(radii, length=coords.shape[1])

        S = transformation.s_matrix(1.0 / radii)
        self.rIndexKD = IndexKD(coords, S)

    def __call__(self, coords):
        mIndexKD = IndexKD(coords, self.rIndexKD.t)
        rIndexKD = self.rIndexKD

        rDists, rIds = rIndexKD.knn(
            mIndexKD.coords, k=1, distance_upper_bound=1)

        mDists, mIds = mIndexKD.knn(
            rIndexKD.coords, k=1, distance_upper_bound=1)

        pairs = []
        for rId in range(len(rIds)):
            if rDists[rId] <= 1:
                if rId == mIds[rIds[rId]]:
                    pairs.append((rIds[rId], rId))

        return np.array(pairs, dtype=int)


class SphereMatcher(Matcher):
    """Find pairs of points. Each point is assigned to all the points
    within a previously defined sphere. Duplicate assignments are possible.

    See Also
    --------
    Matcher

    Examples
    --------

    Create point sets.

    >>> A = np.array([(0, 0), (0, 0.1), (1, 1), (1, 0), (0.5, 0.5), (-1, -2)])
    >>> B = np.array([(0.4, 0.4), (0.2, 0), (0.1, 1.2), (2, 1), (-1.1, -1.2)])

    Match points.

    >>> matcher = SphereMatcher(A, [0.3, 0.2])
    >>> pairs = matcher(B)
    >>> print_rounded(pairs)
    [[4 0]
     [0 1]
     [1 1]]
    >>> print_rounded(A[pairs[:, 0], :] - B[pairs[:, 1], :])
    [[ 0.1  0.1]
     [-0.2  0. ]
     [-0.2  0.1]]

    """

    def __call__(self, coords):
        mIndexKD = IndexKD(coords, self.rIndexKD.t)
        rIndexKD = self.rIndexKD

        pairs = []
        ball_gen = rIndexKD.ball_iter(mIndexKD.coords, 1)
        for mId, rIds in enumerate(ball_gen):
            for rId in rIds:
                pairs.append((rId, mId))

        return np.array(pairs, dtype=int)


class KnnMatcher(Matcher):
    """Find pairs of points. Each point is assigned to `k` closest points
    within a predefined sphere. Duplicate assignents are possible.

    See Also
    --------
    Matcher

    Examples
    --------

    Create coordinates and matcher.

    >>> A = np.array([(0, 0), (0, 0.1), (1, 1), (1, 0), (0.5, 0.5), (-1, -2)])
    >>> B = np.array([(0.4, 0.4), (0.2, 0), (0.1, 1.2), (2, 1), (-1.1, -1.2)])
    >>> matcher = KnnMatcher(A, [0.5, 0.5])

    Try to assign one neighbour.

    >>> pairs = matcher(B)
    >>> print_rounded(pairs)
    [[4 0]
     [0 1]]

    Try to assign two neighbours.

    >>> pairs = matcher(B, k=2)
    >>> print_rounded(pairs)
    [[4 0]
     [0 1]
     [1 1]]

    """

    def __call__(self, coords, k=1):
        """Assign `k` closest points.

        Parameters
        ----------
        k : optional, int
            Number of neighbours to assign.

        See Also
        --------
        Matcher

        """
        if not (isinstance(k, int) and k > 0):
            raise ValueError("'k' needs to be an integer greater zero")
        mIndexKD = IndexKD(coords, self.rIndexKD.t)
        rIndexKD = self.rIndexKD

        pairs = []
        mCoords = mIndexKD.coords
        ball_gen = rIndexKD.knn_iter(mCoords, k, distance_upper_bound=1)
        for mId, (dists, rIds) in enumerate(ball_gen):
            if k == 1:
                dists = [dists]
                rIds = [rIds]
            for dist, rId in zip(dists, rIds):
                if dist <= 1:
                    pairs.append((rId, mId))

        return np.array(pairs, dtype=int)
