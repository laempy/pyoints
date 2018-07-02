"""Registration or alignment of point sets.
"""

import numpy as np

from .. indexkd import IndexKD
from .. import (
    assertion,
    transformation,
    distance,
    nptools,
    assign,
)
from . import rototranslations
#from .rototranslations import find_rototranslations


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
    M : np.matrix(Number, shape = (k+1, k+1)
        Tranformation matrix which maps `B` to `A` with A = `B * M.T`.

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
    A : array_like(Number, shape=(n, k))
        Arrays representing `n` reference points with `k` dimensions.

    Returns
    -------
    M : numpy.matrix(float, shape=(k+1, k+1))
        Roto-translation matrix which maps `B` to `A` with `A = B * M.T`.

    References
    ----------
    TODO: references
    http://nghiaho.com/?page_id=671
    http://nghiaho.com/?page_id=671
    # http://nghiaho.com/uploads/code/rigid_transform_3D.py_
    # Zhang_2016a

    Examples
    --------

    >>> T = transformation.matrix(t=[3, 5], r=0.3)
    >>> A = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    >>> B = transformation.transform(A, T)

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


def icp(coords_dict,
        radii,
        T_dict={},
        assign_class=assign.KnnMatcher,
        max_iter=10):
    """Implementation of the Iterative Closest Point algorithm with multiple
    point set support.

    Paramerters
    -----------
    coords_dict : dict
        Dictionary of point sets with `k` dimensions.
    radii :

    m_dict : dict of array_like(Number, shape=(k+1, k+1))
        Dictionary of initial transformation matrices.

    Returns
    -------
    dict of LocalSystem(Number, shape=(k+1, k+1))
        Dictionary of transformation matices.

    Examples
    --------

    >>> A = np.array([
    ...     (0.5, 0.5), (0, 0), (0, -0.1), (1.3, 1), (1, 0), (-1, -2)
    ... ])
    >>> B = np.array([(0.4, 0.5), (0.3, 0), (1, 1), (2, 1), (-1, -2)])

    >>> coords_dict = {'A': A, 'B': B}
    >>> radii = (0.25, 0.25)
    >>> T = icp(coords_dict, radii, max_iter=10)

    Transform coordinates.

    >>> tA = T['A'].to_local(A)
    >>> tB = T['B'].to_local(B)

    >>> print(np.round(tA, 2))
    [[ 0.43  0.51]
     [-0.06  0.  ]
     [-0.06 -0.1 ]
     [ 1.22  1.03]
     [ 0.94  0.02]
     [-1.02 -2.02]]
    >>> print(np.round(tB, 2))
    [[ 0.47  0.5 ]
     [ 0.35 -0.  ]
     [ 1.08  0.98]
     [ 2.08  0.95]
     [-1.   -1.97]]

    Find matches and compare RMSE

    >>> matcher = assign.KnnMatcher(tA, radii)
    >>> pairs = matcher(tB)

    >>> rmse = distance.rmse(A[pairs[:, 0], :], B[pairs[:, 1], :])
    >>> print(np.round(rmse, 3))
    0.183

    >>> rmse = distance.rmse(tA[pairs[:, 0], :], tB[pairs[:, 1], :])
    >>> print(np.round(rmse, 3))
    0.094

    """

    # prepare input

    if not isinstance(coords_dict, dict):
        raise TypeError("'coords_dict' needs to be a dictionary")
    if len(coords_dict) < 2:
        raise ValueError("at least two point sets are required")
    if not isinstance(T_dict, dict):
        raise TypeError("'T_dict' needs to be a dictionary")

    if not hasattr(assign_class, '__call__'):
        raise TypeError("'assign_class' must be a callable object")

    radii = assertion.ensure_numvector(radii)
    dim = len(radii)

    for key in coords_dict:
        coords_dict[key] = assertion.ensure_coords(coords_dict[key], dim=dim)

    for key in coords_dict:
        if key not in T_dict.keys():
            T_dict[key] = transformation.i_matrix(dim)
        else:
            T_dict[key] = assertion.ensure_tmatrix(T_dict[key])

    # ICP algorithm
    for num_iter in range(max_iter):
        pairs_dict = {}
        for keyA in coords_dict:
            pairs_dict[keyA] = {}
            coordsA = transformation.transform(
                coords_dict[keyA], T_dict[keyA])
            matcher = assign_class(coordsA, radii)

            for keyB in coords_dict:
                if keyB != keyA:

                    coordsB = transformation.transform(
                        coords_dict[keyB], T_dict[keyB])

                    pairs = matcher(coordsB)
                    if len(pairs) > 0:
                        dist = distance.dist(
                            coordsA[pairs[:, 0], :], coordsB[pairs[:, 1], :])
                        w = distance.idw(dist, p=2)
                    else:
                        w = []
                    pairs_dict[keyA][keyB] = (pairs, w)

        # find roto-translation matrices
        T_dict = rototranslations.find_rototranslations(
            coords_dict,
            pairs_dict
        )

        # termination
        if num_iter == 0:
            pairs_old = pairs_dict
        else:
            change = False
            for keyA in pairs_dict:
                for keyB in pairs_dict:
                    if not keyB == keyA:
                        p_old = pairs_old[keyA][keyB][0]
                        p_new = pairs_dict[keyA][keyB][0]
                        if not np.array_equal(p_new, p_old):
                            change = True
            if not change:
                break
            pairs_old = pairs_dict

    return T_dict
