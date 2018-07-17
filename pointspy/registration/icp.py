"""Implementation of the Iterative Closest Point Algorithm"""

import numpy as np

from . import rototranslations
from .. import (
    assertion,
    transformation,
    distance,
    assign,
    fit,
)


def icp(coords_dict,
        radii,
        normals_dict={},
        pairs_dict={},
        T_dict={},
        assign_class=assign.KnnMatcher,
        max_iter=10,
        **assign_parameters):
    """Implementation of the Iterative Closest Point algorithm with multiple
    point set support.

    Paramerters
    -----------
    TODO: parameters
    coords_dict : dict
        Dictionary of point sets with `k` dimensions.
    radii :

    m_dict : dict of array_like(Number, shape=(k+1, k+1))
        Dictionary of initial transformation matrices.

    Returns
    -------
    dict of LocalSystem(Number, shape=(k+1, k+1))
        Dictionary of transformation matices.

    References
    ----------
    TODO: Ref

    Examples
    --------

    >>> A = np.array([
    ...     (0.5, 0.5), (0, 0), (0, -0.1), (1.3, 1), (1, 0), (-1, -2)
    ... ])
    >>> B = np.array([(0.4, 0.5), (0.3, 0), (1, 1), (2, 1), (-1, -2)])

    >>> coords_dict = {'A': A, 'B': B}
    >>> radii = (0.25, 0.25)

    Standard ICP.

    >>> T, pairs = icp(coords_dict, radii, max_iter=10, k=1)

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

    NICP which uses normals for weighting.

    >>> normals_r = 1.5
    >>> normals_dict = {
    ...     'A': fit.fit_normals(A, normals_r),
    ...     'B': fit.fit_normals(B, normals_r)
    ... }
    >>> print(normals_dict)

    >>> T, pairs = icp(coords_dict, radii, normals_dict=normals_dict)

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

    """
    # prepare input
    if not isinstance(coords_dict, dict):
        raise TypeError("'coords_dict' needs to be a dictionary")
    if len(coords_dict) < 2:
        raise ValueError("at least two point sets are required")
    if not isinstance(T_dict, dict):
        raise TypeError("'T_dict' needs to be a dictionary")
    if not isinstance(pairs_dict, dict):
        raise TypeError("'pairs_dict' needs to be a dictionary")
    if not hasattr(assign_class, '__call__'):
        raise TypeError("'assign_class' must be a callable object")

    radii = assertion.ensure_numvector(radii)
    dim = len(radii)

    # double check coordinate format
    for key in coords_dict:
        coords_dict[key] = assertion.ensure_coords(coords_dict[key], dim=dim)

    # check normals
    if not isinstance(normals_dict, dict):
        raise TypeError("'normals_dict' needs to be a dictionary")
    if len(normals_dict) > 0:
        for key in coords_dict:
            if key in normals_dict:
                normals_dict[key] = assertion.ensure_coords(normals_dict[key], dim=dim)
            else:
                raise ValueError("missing normals for '%s'" % key)


    # Define initial transformation matrices
    if len(pairs_dict) > 0:
        T_dict = rototranslations.find_rototranslations(
            coords_dict, pairs_dict)
    else:
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

                    pairs = matcher(coordsB, **assign_parameters)
                    if len(pairs) > 0:
                        if len(normals_dict) > 0:
                            dists = distance.dist(
                                normals_dict[keyA][pairs[:, 0], :],
                                normals_dict[keyB][pairs[:, 1], :]
                            )
                            w = distance.idw(dists, p=2)
                        else:
                            dists = distance.dist(
                                coordsA[pairs[:, 0], :],
                                coordsB[pairs[:, 1], :]
                            )
                            w = distance.idw(dists, p=2)
                    else:
                        w = []
                    pairs_dict[keyA][keyB] = (pairs, w)

        # find roto-translation matrices
        T_dict = rototranslations.find_rototranslations(
            coords_dict, pairs_dict)

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