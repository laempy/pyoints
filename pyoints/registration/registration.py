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
"""Registration or alignment of point sets.
"""

import numpy as np

from .. import (
    assertion,
    transformation,
)
from ..misc import print_rounded


def find_transformation(A, B):
    """Finds the optimal (non-rigid) transformation matrix `M` between two
    point sets. Each point of point set `A` is associated with exactly one
    point in point set `B`.

    Parameters
    ----------
    A : array_like(Number, shape=(n, k))
        Array representing n reference points with k dimensions.
    B : array_like(Number, shape=(n, k))
        Array representing n points with k dimensions.

    Returns
    -------
    M : np.matrix(Number, shape=(k+1, k+1))
        Tranformation matrix which maps `B` to `A` with A = `B * M.T`.

    See Also
    --------
    find_rototranslation

    """
    b = transformation.homogenious(A)
    mA = transformation.homogenious(B)

    if not b.shape == mA.shape:
        raise ValueError('dimensions do not match')

    x = np.linalg.lstsq(mA, b, rcond=None)[0]
    M = x.T

    return M


def find_rototranslation(A, B):
    """Finds the optimal roto-translation matrix `M` between two point sets.
    Each point of point set `A` is associated with exactly one point in point
    set `B`.

    Parameters
    ----------
    A,B : array_like(Number, shape=(n, k))
        Arrays representing `n` corresponding points with `k` dimensions.

    Returns
    -------
    M : numpy.matrix(float, shape=(k+1, k+1))
        Roto-translation matrix to map `B` to `A` with `A = B * M.T`.

    Notes
    -----
    Implements the registration algorithm of Besl and McKay (1992) [1]. The
    idea has been taken from Nghia Ho (2013) [2]. Code of [2] has been
    generalized to `k` dimensional space.

    References
    ----------

    [1] P. J. Besl and N. D. McKay (1992): "A Method for Registration of 3-D
    Shapes", IEEE Transactions on Pattern Analysis and Machine Intelligence,
    Institute of Electrical and Electronics Engineers (IEEE), vol. 14,
    pp. 239-256.

    [2] Nghia Ho (2013): "Finding optimal rotation and translation between
    corresponding 3D points", URL http:\/\/nghiaho.com/\?page\_id=671.

    [3] Nghia Ho (2013): "Finding optimal rotation and translation between
    corresponding 3D points", URL
    http:\/\/nghiaho.com/uploads/code/rigid\_transform\_3D.py\_.

    Examples
    --------

    Creates similar, but shifted and rotated point sets.

    >>> A = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    >>> B = transformation.transform(A, transformation.matrix(t=[3, 5], r=0.3))

    Finds roto-translation.

    >>> M = find_rototranslation(A, B)

    >>> C = transformation.transform(B, M, inverse=False)
    >>> print_rounded(C, 2)
    [[0. 0.]
     [0. 1.]
     [1. 1.]
     [1. 0.]]

    """
    A = assertion.ensure_coords(A)
    B = assertion.ensure_coords(B)

    if not A.shape == B.shape:
        raise ValueError("coordinate dimensions do not match")

    cA = A.mean(0)
    cB = B.mean(0)
    mA = transformation.homogenious(A - cA, value=0)
    mB = transformation.homogenious(B - cB, value=0)

    # Find rotation matrix
    H = mA.T @ mB
    U, S, V = np.linalg.svd(H)
    R = U @ V

    # reflection case
    if np.linalg.det(R) < 0:
        R[-1, :] = -R[-1, :]
        # TODO test

    # Create transformation matrix
    T1 = transformation.t_matrix(cA)
    T2 = transformation.t_matrix(-cB)
    M = T1 @ R @ T2

    return transformation.LocalSystem(M)
