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
"""Implementation of the Iterative Closest Point Algorithm.
"""

import numpy as np
from numbers import Number

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
        points of `k` dimensions. The length of `radii` is equal to `2 * k`.
        If point normals shall also be used to find point pairs, the length of
        `radii` is `k`.
    assign_class : optional, callable class
        Class which assigns pairs of points.
    max_iter : optional, positive int
        Maximum number of iterations.
    update_normals : bool
        Indicates whether or not to also transform the normals.
    \*\*assign_parameters
        Parameters passed to `assign_class`.

    Notes
    -----
    A modified variant of the originally ICP algorithm presented by Besl and 
    McKay (1992) [1].

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
    >>> T, pairs, report = icp(coords_dict, weights=weights)

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

    Find matches and compare RMSE (Root Mean Squared Error).

    >>> matcher = assign.KnnMatcher(tA, radii)
    >>> pairs = matcher(tB)

    >>> rmse = distance.rmse(A[pairs[:, 0], :], B[pairs[:, 1], :])
    >>> print(np.round(rmse, 3))
    0.183

    >>> rmse = distance.rmse(tA[pairs[:, 0], :], tB[pairs[:, 1], :])
    >>> print(np.round(rmse, 3))
    0.09

    ICP also takes advantage of normals (NICP).

    >>> from pyoints.normals import fit_normals
    >>> normals_r = 1.5
    >>> normals_dict = {
    ...     'A': fit_normals(A, normals_r),
    ...     'B': fit_normals(B, normals_r)
    ... }
    >>> radii = (0.25, 0.25, 0.3, 0.3)

    >>> nicp = ICP(radii, max_iter=10, k=1)
    >>> T, pairs, report = nicp(coords_dict, normals_dict=normals_dict)

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
                 max_iter=50,
                 max_change_ratio=0.01,
                 assign_class=assign.KnnMatcher,
                 update_normals=False,
                 **assign_parameters):

        if not callable(assign_class):
            raise TypeError("'assign_class' must be a callable")
        if not (isinstance(max_iter, int) and max_iter >= 0):
            raise ValueError("'max_iter' must be an integer greater zero")
        if not isinstance(update_normals, bool):
            raise TypeError("'update_normals' must be boolean")
        if not (isinstance(max_change_ratio, Number) and max_change_ratio > 0):
            raise ValueError(
                "'max_change_ratio' must be a number greater zero")

        self._assign_class = assign_class
        self._radii = assertion.ensure_numvector(radii, min_length=2)
        self._max_iter = max_iter
        self._max_change_ratio = max_change_ratio
        self._update_normals = update_normals
        self._assign_parameters = assign_parameters

    def __call__(
            self,
            coords_dict,
            normals_dict={},
            pairs_dict={},
            T_dict={},
            overlap_dict={},
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
            `T_dict` will be overwritten.
        overlap_dict : optional, dict of list(str)
            Dictionary specifying which point clouds overlap.
        weights : optional, array_like(Number)
            Weights passed to `find_rototranslations`.

        Returns
        -------
        T_dict : dict of array_like(int, shape=(k+1, k+1))
            Desired dictionary of transformation matrices.
        pairs_dict : dict of array_like(int, shape=(m, 2))
            Desired dictionary of point pairs.
        report : dict
            Report to evaluate the quality of the results.

        See Also
        --------
        find_rototranslations

        """
        # validate input
        coords_dict, dim = _ensure_coords_dict(coords_dict)
        overlap_dict = _ensure_overlap_dict(
            coords_dict, overlap_dict)
        normals_dict = _ensure_normals_dict(normals_dict, coords_dict)
        T_dict = _ensure_T_dict(T_dict, coords_dict, pairs_dict, weights)

        # check radii
        if len(normals_dict) > 0:
            if not len(self._radii) == 2 * dim:
                m = "NICP requires %i radii, got %i"
                raise ValueError(m % (2 * dim, len(self._radii)))
        else:
            if not len(self._radii) == dim:
                m = "ICP requires %i radii, got %i" % (dim, len(self._radii))
                raise ValueError(m % (2 * dim, len(self._radii)))

        max_change = distance.norm(self._radii[:dim]) * self._max_change_ratio

        # ICP algorithm
        report = {'RMSE': []}
        for num_iter in range(self._max_iter):

            # assign pairs
            pairs_dict = {}
            for keyA in overlap_dict:
                pairs_dict[keyA] = {}

                A = _get_nCoords(
                    coords_dict,
                    normals_dict,
                    T_dict, keyA,
                    self._update_normals
                )
                matcher = self._assign_class(A, self._radii)

                for keyB in overlap_dict[keyA]:

                    B = _get_nCoords(
                        coords_dict,
                        normals_dict,
                        T_dict, keyB,
                        self._update_normals
                    )
                    pairs = matcher(B, **self._assign_parameters)

                    if len(pairs) > 0:
                        dists = distance.dist(
                            A[pairs[:, 0], :dim],
                            B[pairs[:, 1], :dim],
                        )
                        w = distance.idw(dists, p=2)
                    else:
                        w = []
                    pairs_dict[keyA][keyB] = (pairs, w)

            # find roto-translation matrices
            T_dict_new = rototranslations.find_rototranslations(
                coords_dict, pairs_dict, weights=weights)

            # take a look at the residuals between before and after
            rmse = _get_change_rmse(coords_dict, T_dict, T_dict_new)
            report['RMSE'].append(rmse)
            if rmse <= max_change:
                break

            T_dict = T_dict_new

        return T_dict, pairs_dict, report


def _get_change_rmse(coords_dict, T_dict_old, T_dict_new):
    rmse_dict = {}
    for key, coords in coords_dict.items():
        coords_old = transformation.transform(coords, T_dict_old[key])
        coords_new = transformation.transform(coords, T_dict_new[key])
        rmse_dict[key] = distance.rmse(coords_new, coords_old)
    return np.max(list(rmse_dict.values()))


def _get_nCoords(coords_dict, normals_dict, T_dict, key, update_normals):
    nCoords = coords_dict[key]
    T = T_dict[key]
    nCoords = transformation.transform(coords_dict[key], T)

    if len(normals_dict) > 0:
        if update_normals:
            R = transformation.r_matrix(transformation.decomposition(T)[1])
            normals = transformation.transform(normals_dict[key], R)
        else:
            normals = normals_dict[key]
        nCoords = np.hstack((nCoords, normals))
    return nCoords


def _ensure_coords_dict(coords_dict):
    if not isinstance(coords_dict, dict):
        raise TypeError("'coords_dict' needs to be a dictionary")

    dim = None
    out_coords_dict = {}
    for key in coords_dict:
        if dim is None:
            out_coords_dict[key] = assertion.ensure_coords(coords_dict[key])
            dim = out_coords_dict[key].shape[1]
        else:
            out_coords_dict[key] = assertion.ensure_coords(
                coords_dict[key], dim=dim)
    return out_coords_dict, dim


def _ensure_overlap_dict(coords_dict, overlap_dict):
    if not isinstance(overlap_dict, dict):
        raise TypeError("'cloud_pairs_dict' needs to be a dictionary")

    out_dict = {}
    if len(overlap_dict) == 0:
        for keyA in coords_dict:
            out_dict[keyA] = [keyB for keyB in coords_dict if keyB is not keyA]
    else:
        # check dict
        for keyA in coords_dict:
            if keyA not in overlap_dict:
                raise ValueError("missing key")
            if not isinstance(overlap_dict[keyA], (list, tuple)):
                raise ValueError("tuple or list required")
            for keyB in overlap_dict[keyA]:
                if keyB not in coords_dict:
                    raise ValueError("unknown key")
            out_dict[keyA] = overlap_dict[keyA]
    return out_dict


def _ensure_normals_dict(normals_dict, coords_dict,):
    if not isinstance(normals_dict, dict):
        raise TypeError("'normals_dict' needs to be a dictionary")
    out_normals_dict = {}
    if len(normals_dict) > 0:
        for key in coords_dict:
            dim = coords_dict[key].shape[1]
            if key in normals_dict:
                out_normals_dict[key] = assertion.ensure_coords(
                    normals_dict[key], dim=dim)
            else:
                raise ValueError("missing normals for '%s'" % key)
    return out_normals_dict


def _ensure_T_dict(T_dict, coords_dict, pairs_dict, weights):
    if not isinstance(T_dict, dict):
        raise TypeError("'T_dict' needs to be a dictionary")
    if not isinstance(pairs_dict, dict):
        raise TypeError("'pairs_dict' needs to be a dictionary")

    out_T_dict = {}
    if len(T_dict) == 0 and len(pairs_dict) > 0:
        out_T_dict = rototranslations.find_rototranslations(
            coords_dict, pairs_dict, weights=weights)
    else:
        for key in coords_dict:
            if key in T_dict.keys():
                out_T_dict[key] = assertion.ensure_tmatrix(T_dict[key])
            else:
                dim = coords_dict[key].shape[1]
                out_T_dict[key] = transformation.i_matrix(dim)
    return out_T_dict
