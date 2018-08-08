# BEGIN OF LICENSE NOTE
# This file is part of PoYnts.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Registration or alignment of point sets.
"""

import numpy as np

from .. import (
    assertion,
    transformation,
)


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
    M = np.matrix(x).T

    return M


def find_rototranslation(A, B):
    """Finds the optimal roto-translation matrix `M` between two point sets.
    Each point of point set `A` is associated with exactly one point in point
    set `B`.

    Parameters
    ----------
    A,B : array_like(Number, shape=(n, k))
        Arrays representing `n` reference points with `k` dimensions.

    Returns
    -------
    M : numpy.matrix(float, shape=(k+1, k+1))
        Roto-translation matrix which maps `B` to `A` with `A = B * M.T`.

    Notes
    -----
    Implements the algorithm of Besl and McKay (1992) [1]. Idea has been taken
    from  Nghia Ho (2013) [2]. Code [2] has been adapted generalized to `k`
    dimensional case.

    References
    ----------

    [1] P. J. Besl and N. D. McKay (1992): "A Method for Registration of 3-D
    Shapes", IEEE Transactions on Pattern Analysis and Machine Intelligence,
    Institute of Electrical and Electronics Engineers (IEEE), vol. 14,
    pp. 239-256.

    [2] Nghia Ho (2013): "Finding optimal rotation and translation between
    corresponding 3D points", URL http://nghiaho.com/?page\_id=671.

    [3] Nghia Ho (2013): "Finding optimal rotation and translation between
    corresponding 3D points", URL
    http://nghiaho.com/uploads/code/rigid_transform\_3D.py\_.

    Examples
    --------

    Create similar, but shifted and rotated point sets.

    >>> A = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    >>> B = transformation.transform(A, transformation.matrix(t=[3, 5], r=0.3))

    Find roto-translation.

    >>> M = find_rototranslation(A, B)

    >>> C = transformation.transform(B, M, inverse=False)
    >>> print(np.round(C, 2))
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
    mA = np.matrix(transformation.homogenious(A - cA, value=0))
    mB = np.matrix(transformation.homogenious(B - cB, value=0))

    # Find rotation matrix
    H = mA.T * mB
    U, S, V = np.linalg.svd(H)
    R = U * V

    # reflection case
    if np.linalg.det(R) < 0:
        R[-1, :] = -R[-1, :]
        # TODO test

    # Create transformation matrix
    T1 = transformation.t_matrix(cA)
    T2 = transformation.t_matrix(-cB)
    M = T1 * R * T2

    return transformation.LocalSystem(M)
