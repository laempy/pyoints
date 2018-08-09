# BEGIN OF LICENSE NOTE
# This file is part of Pointspy.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
#
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Implementation of the Iterative Closest Point Algorithm.
"""

import numpy as np

from . import rototranslations
from .. import (
    assertion,
    transformation,
    distance,
    assign,
)


class ICP:
    """Implementation of a variant of the Iterative Closest Point algorithm [1]
    with support of multiple point sets and point normals.

    Parameters
    ----------
    radii : array_like(Number, shape=(s))
        Maximum distances in each coordinate dimension to assign corresponding
        points of `k` dimensions. The length of `radii` is equal to `2 * k`
        if point normals shall also be used to find point pairs, `k`
        otherwise.
    assign_class : optional, callable class
        Class which assigns pairs of points.
    max_iter : optional, positive int
        Maximum number of iterations.

    Notes
    -----
    An own variant of the originally ICP algorithm presented by Besl and McKay
    (1992) [1].

    References
    ----------

    [1] P.J. Besl and N.D. McKay (1992): "A Method for Registration of 3-D
    Shapes", IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. 14 (2): 239-256.

    Examples
    --------

    Create corresponding point sets.

    >>> A = np.array([
    ...     (0.5, 0.5), (0, 0), (0, -0.1), (1.3, 1), (1, 0), (-1, -2)
    ... ])
    >>> B = np.array([(0.4, 0.5), (0.3, 0), (1, 1), (2, 1), (-1, -2)])

    Standard ICP.

    >>> coords_dict = {'A': A, 'B': B}
    >>> radii = (0.25, 0.25)
    >>> weights = {'A': [1, 1, 1]}
    >>> icp = ICP(radii, max_iter=10, k=1)
    >>> T, pairs = icp(coords_dict, weights=weights)

    >>> tA = T['A'].to_local(A)
    >>> tB = T['B'].to_local(B)

    >>> print(np.round(tA, 2))
    [[ 0.5  0.5]
     [ 0.   0. ]
     [-0.  -0.1]
     [ 1.3  1. ]
     [ 1.  -0. ]
     [-1.  -2. ]]
    >>> print(np.round(tB, 2))
    [[ 0.56  0.48]
     [ 0.43 -0.01]
     [ 1.19  0.95]
     [ 2.18  0.89]
     [-0.97 -1.94]]

    Find matches and compare RMSE.

    >>> matcher = assign.KnnMatcher(tA, radii)
    >>> pairs = matcher(tB)

    >>> rmse = distance.rmse(A[pairs[:, 0], :], B[pairs[:, 1], :])
    >>> print(np.round(rmse, 3))
    0.183

    >>> rmse = distance.rmse(tA[pairs[:, 0], :], tB[pairs[:, 1], :])
    >>> print(np.round(rmse, 3))
    0.09

    ICP also take advantage of normals (NICP).

    >>> from pointspy import fit
    >>> normals_r = 1.5
    >>> normals_dict = {
    ...     'A': fit.fit_normals(A, normals_r),
    ...     'B': fit.fit_normals(B, normals_r)
    ... }
    >>> radii = (0.25, 0.25, 0.3, 0.3)

    >>> nicp = ICP(radii, max_iter=10, k=1)
    >>> T, pairs = nicp(coords_dict, normals_dict=normals_dict)

    >>> tA = T['A'].to_local(A)
    >>> print(np.round(tA, 2))
    [[ 0.5  0.5]
     [ 0.   0. ]
     [ 0.  -0.1]
     [ 1.3  1. ]
     [ 1.   0. ]
     [-1.  -2. ]]

    >>> tB = T['B'].to_local(B)
    >>> print(np.round(tB, 2))
    [[ 0.4  0.5]
     [ 0.3  0. ]
     [ 1.   1. ]
     [ 2.   1. ]
     [-1.  -2. ]]

    """

    def __init__(self,
                 radii,
                 max_iter=10,
                 assign_class=assign.KnnMatcher,
                 **assign_parameters):

        if not hasattr(assign_class, '__call__'):
            raise TypeError("'assign_class' must be a callable object")
        if not (isinstance(max_iter, int) and max_iter >= 0):
            raise ValueError("'max_iter' needs to be an integer greater zero")

        self._assign_class = assign_class
        self._radii = assertion.ensure_numvector(radii, min_length=2)
        self._max_iter = max_iter
        self._assign_parameters = assign_parameters

    def __call__(self,
                 coords_dict,
                 sampleids_dict={},
                 normals_dict={},
                 pairs_dict={},
                 T_dict={},
                 weights=None):
        """Calls the ICP algorithm to align multiple point sets.

        Parameters
        ----------
        coords_dict : dict of array_like(Number, shape=(n, k))
            Dictionary of point sets with `k` dimensions.
        normals_dict : optional, dict of array_like(Number, shape=(n, k))
            Dictionary of corresponding point normals.
        pairs_dict : optional, dict of array_like(int, shape=(m, 2))
            Dictionary of point pairs.
        T_dict : optional, dict of array_like(int, shape=(k+1, k+1))
            Dictionary of transformation matrices. If `pairs_dict` is provided,
            `T_dict` will be calculated automatically.

        Returns
        -------
        T_dict : dict of array_like(int, shape=(k+1, k+1))
            Desired dictionary of transformation matrices.
        pairs_dict : dict of array_like(int, shape=(m, 2))
            Desired dictionary of point pairs.

        """
        # validate input
        coords_dict, dim = _ensure_coords_dict(coords_dict)
        sampleids_dict = _ensure_sampleids_dict(
            sampleids_dict, coords_dict)
        normals_dict = _ensure_normals_dict(normals_dict, coords_dict)
        T_dict = _ensure_T_dict(T_dict, coords_dict, pairs_dict, weights)

        # check radii
        if len(normals_dict) > 0:
            if not len(self._radii) == 2 * dim:
                m = "NICP requires %i radii got %i"
                raise ValueError(m % (2 * dim, len(self._radii)))
        else:
            if not len(self._radii) == dim:
                m = "ICP requires %i radii got %i" % (dim, len(self._radii))
                raise ValueError(m % (2 * dim, len(self._radii)))

        # ICP algorithm
        for num_iter in range(self._max_iter):

            # assign pairs
            pairs_dict = {}
            for keyA in coords_dict:
                pairs_dict[keyA] = {}

                A = _get_nCoords(
                    coords_dict, normals_dict, T_dict, keyA)
                matcher = self._assign_class(A, self._radii)

                for keyB in coords_dict:
                    if keyB != keyA:

                        B = _get_nCoords(
                            coords_dict, normals_dict, T_dict, keyB)
                        sids = sampleids_dict[keyB]
                        pairs = matcher(B[sids, :], **self._assign_parameters)

                        if len(pairs) > 0:
                            pairs[:, 1] = sids[pairs[:, 1]]
                            dists = distance.dist(
                                A[pairs[:, 0], :dim], B[pairs[:, 1], :dim])
                            w = distance.idw(dists, p=2)
                        else:
                            w = []
                        pairs_dict[keyA][keyB] = (pairs, w)

            # find roto-translation matrices
            T_dict = rototranslations.find_rototranslations(
                coords_dict, pairs_dict, weights=weights)

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

        return T_dict, pairs_dict


def _get_nCoords(coords_dict, normals_dict, T_dict, key):
    nCoords = coords_dict[key]
    T = T_dict[key]
    nCoords = transformation.transform(coords_dict[key], T)

    if len(normals_dict) > 0:
        # update normal orientation
        #R = transformation.r_matrix(transformation.decomposition(T)[1])
        #normals = transformation.transform(normals_dict[key], R)
        normals = normals_dict[key]
        nCoords = np.hstack((nCoords, normals))
    return nCoords


def _ensure_coords_dict(coords_dict):
    if not isinstance(coords_dict, dict):
        raise TypeError("'coords_dict' needs to be a dictionary")

    dim = None
    for key in coords_dict:
        if dim is None:
            coords_dict[key] = assertion.ensure_coords(coords_dict[key])
            dim = coords_dict[key].shape[1]
        coords_dict[key] = assertion.ensure_coords(
            coords_dict[key], dim=dim)
    return coords_dict, dim


def _ensure_normals_dict(normals_dict, coords_dict):
    if not isinstance(normals_dict, dict):
        raise TypeError("'normals_dict' needs to be a dictionary")
    if len(normals_dict) > 0:
        for key in coords_dict:
            dim = coords_dict[key].shape[1]
            if key in normals_dict:
                normals_dict[key] = assertion.ensure_coords(
                    normals_dict[key], dim=dim)
            else:
                raise ValueError("missing normals for '%s'" % key)
    return normals_dict


def _ensure_sampleids_dict(sampleids_dict, coords_dict):
    if not isinstance(sampleids_dict, dict):
        raise TypeError("'sampleids_dict' needs to be a dictionary")
    for key in coords_dict:
        n = len(coords_dict[key])
        if key in sampleids_dict:
            sampleids_dict[key] = assertion.ensure_indices(
                sampleids_dict[key], max_value=n)
        else:
            sampleids_dict[key] = np.arange(n)
    return sampleids_dict


def _ensure_T_dict(T_dict, coords_dict, pairs_dict, weights):
    if not isinstance(T_dict, dict):
        raise TypeError("'T_dict' needs to be a dictionary")
    if not isinstance(pairs_dict, dict):
        raise TypeError("'pairs_dict' needs to be a dictionary")
    if len(T_dict) > 0 and len(pairs_dict) > 0:
        raise ValueError("please specifiy either 'T_dict' or 'pairs_dict'")

    if len(T_dict) == 0 and len(pairs_dict) > 0:
        T_dict = rototranslations.find_rototranslations(
            coords_dict, pairs_dict, weights=weights)
    else:
        for key in coords_dict:
            if key in T_dict.keys():
                T_dict[key] = assertion.ensure_tmatrix(T_dict[key])
            else:
                dim = coords_dict[key].shape[1]
                T_dict[key] = transformation.i_matrix(dim)
    return T_dict
