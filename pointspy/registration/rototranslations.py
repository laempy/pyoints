"""Find roto-translation matrices of multiple point sets.
"""

import numpy as np

from .. import (
    assertion,
    transformation,
    nptools,
)


def find_rototranslations(coords_dict, pairs_dict, weights=None):
    """Find the optimal roto-translation matrices between multiple
    point sets using pairs of points. The algorithm assumes infinitissimal
    rotations between the point sets.

    Parameters
    ----------
    coords_dict: dict of array_like(int, shape=(n, k))
        Dictionary of point sets with `k` dimensions.
    pairs_dict : dict of array_like(int, shape=(m, 2))
        Dictionary of point pairs.
    weights : optional, dict or list or int.
        Try to keep the original location and orientation by weighting. Each
        point set can be weighted by a list of `2 * k` values. The first `k`
        values represent the weighting factors for location. The last `k`
        values represent the weighting factors for orientation (angles).
        The weights can be provided for each point set individially in form
        of a dictionary weights. If not provided, weights are set to zero.

    Examples
    --------

    2D coordinates.

    >>> coordsA = [(-1, -2), (-1, 2), (1, 2), (1, -2), (5, 10)]
    >>> T = transformation.matrix(
    ...     t=[100000, 4000],
    ...     r=0.04,
    ... )
    >>> coordsB = transformation.transform(coordsA, T)

    >>> coords_dict = {'A': coordsA, 'B': coordsB}
    >>> pairs_dict = { 'A': { 'B': [(0, 0), (1, 1), (2, 2)] } }
    >>> weights = {'A': [1, 1, 1], 'B': [0, 0, 0]}

    >>> res = find_rototranslations(coords_dict, pairs_dict, weights=weights)
    >>> print(list(res.keys()))
    ['A', 'B']
    >>> tA = res['A'].to_local(coords_dict['A'])
    >>> print(np.round(tA, 2))
    [[-1. -2.]
     [-1.  2.]
     [ 1.  2.]
     [ 1. -2.]
     [ 5. 10.]]
    >>> tB = res['B'].to_local(coords_dict['B'])
    >>> print(np.round(tB, 2))
    [[-1. -2.]
     [-1.  2.]
     [ 1.  2.]
     [ 1. -2.]
     [ 5. 10.]]

    3D coordinates.

    >>> coordsA = [(-1, -2, 3), (-1, 2, 4), (1, 2, 5), (1, -2, 6)]
    >>> T = transformation.matrix(
    ...     t=[10000, 20000, 3000],
    ...     r=[0.01, 0.01, -0.002],
    ... )
    >>> coordsB = transformation.transform(coordsA, T)

    >>> coords_dict = {'A': coordsA, 'B': coordsB}
    >>> pairs_dict = { 'A': { 'B': [(0, 0), (1, 1), (2, 2)] } }
    >>> weights = {'A': [1, 1, 1, 1, 1, 1], 'B': [0, 0, 0, 0, 0, 0]}
    >>> res = find_rototranslations(coords_dict, pairs_dict, weights=weights)
    >>> print(list(res.keys()))
    ['A', 'B']
    >>> tA = res['A'].to_local(coords_dict['A'])
    >>> print(np.round(tA, 2))
    [[-1. -2.  3.]
     [-1.  2.  4.]
     [ 1.  2.  5.]
     [ 1. -2.  6.]]
    >>> tB = res['B'].to_local(coords_dict['B'])
    >>> print(np.round(tB, 2))
    [[-1. -2.  3.]
     [-1.  2.  4.]
     [ 1.  2.  5.]
     [ 1. -2.  6.]]

    """
    # prepare input
    dim, ccoords, centers, pairs, w = _prepare_input(
        coords_dict,
        pairs_dict,
        weights
    )

    # get equations
    rA, rB = _build_rototranslation_equations(centers, ccoords, pairs, w)
    oA, oB = _build_location_orientation_equations(centers, w)
    if not len(rB) + len(oB) > 0:
        raise ValueError("At least one equation is needed")

    # solve linear equation system
    mA = np.vstack(rA + oA)
    mB = np.hstack(rB + oB)
    M = np.linalg.lstsq(mA, mB, rcond=None)[0]

    # Extract roto-transformation matrices
    T = _extract_transformations(M, centers)

    return T


def _unknowns(dim):
    if dim == 2:
        return 3
    elif dim == 3:
        return 6
    else:
        raise ValueError("%i dimensions not supported" % dim)


def _equations(coords):
    N, dim = coords.shape
    cols = _unknowns(dim)
    if dim == 2:

        Mx = np.zeros((N, cols))
        Mx[:, 0] = 1  # t_x
        Mx[:, 2] = -coords[:, 1]  # -r

        My = np.zeros((N, cols))
        My[:, 1] = 1  # t_y
        My[:, 2] = coords[:, 0]  # r

        return np.vstack((Mx, My))

    elif dim == 3:

        Mx = np.zeros((N, cols))
        Mx[:, 0] = 1  # t_x
        Mx[:, 4] = coords[:, 2]  # z
        Mx[:, 5] = -coords[:, 1]  # -y

        My = np.zeros((N, cols))
        My[:, 1] = 1  # t_y
        My[:, 3] = -coords[:, 2]  # -z
        My[:, 5] = coords[:, 0]  # x

        Mz = np.zeros((N, cols))
        Mz[:, 2] = 1  # t_z
        Mz[:, 3] = coords[:, 1]  # y
        Mz[:, 4] = -coords[:, 0]  # -x

        return np.vstack((Mx, My, Mz))
    else:
        raise ValueError("%i dimensions are not supported yet" % dim)


def _build_rototranslation_equations(centers, ccoords, wpairs, weights):
    # build linear equation system mA = mB * M
    dim = ccoords[list(ccoords.keys())[0]].shape[1]
    unknowns = _unknowns(dim)
    k = len(ccoords)
    mA = []
    mB = []
    for iA, keyA in enumerate(ccoords):
        if keyA in wpairs:
            for iB, keyB in enumerate(ccoords):
                if keyB in wpairs[keyA]:

                    # get pairs of points
                    p = wpairs[keyA][keyB]
                    A = ccoords[keyA][p.A, :]
                    B = ccoords[keyB][p.B, :]

                    # set equations
                    equations_A = _equations(A)
                    equations_B = _equations(B)
                    a = np.zeros((A.shape[0] * dim, k * unknowns))

                    a[:, iA * unknowns:(iA + 1) * unknowns] = equations_A
                    a[:, iB * unknowns:(iB + 1) * unknowns] = -equations_B
                    b = B.T.flatten() - A.T.flatten()

                    # weighting
                    w = np.tile(p.weights, dim)
                    a = (a.T * w).T
                    b = b * w

                    mA.append(a)
                    mB.append(b)

    return mA, mB


def _build_location_orientation_equations(centers, weights):
    # try to keep the original locations and orientations
    k = len(centers)
    dim = len(centers[list(centers.keys())[0]])
    cols = _unknowns(dim)
    mA = []
    mB = []
    for i, key in enumerate(centers):
        if key in weights:

            a = np.eye(cols, k * cols, k=i * cols)
            b = np.zeros(cols)
            b[:dim] = centers[key]

            # TODO weights correct
            w = weights[key]
            a = (a.T * w).T
            b = b * w

            mA.append(a)
            mB.append(b)

    return mA, mB


def _extract_transformations(M, centers):
    dim = len(centers[list(centers.keys())[0]])
    unknowns = _unknowns(dim)
    res = {}
    for i, key in enumerate(centers):
        t = M[i * unknowns:i * unknowns + dim]
        r = M[i * unknowns + dim:(i + 1) * unknowns]

        T0 = transformation.t_matrix(-centers[key])  # mean centering
        T1 = transformation.t_matrix(t)

        R = transformation.r_matrix(r)
        res[key] = T1 * (R * T0)

    return res


def _prepare_input(coords_dict, pairs_dict, weights):

    # type validation
    if not isinstance(coords_dict, dict):
        raise TypeError("'coords_dict' of type 'dict' required")
    if not isinstance(pairs_dict, dict):
        raise TypeError("'pairs_dict' of type 'dict' required")
    for key in pairs_dict:
        if key not in coords_dict:
            m = "key '%s' of 'pairs_dict' not found in 'coords_dict'" % key
            raise ValueError(m)

    # number of point clouds
    k = len(coords_dict)
    if k < 2:
        raise ValueError("at least 2 point sets required")

    # derive centered coordinates
    dim = None
    centers_dict = {}
    ccoords_dict = {}
    for key in coords_dict:
        if dim is None:
            coords = assertion.ensure_coords(coords_dict[key])
            dim = coords.shape[1]
        else:
            coords = assertion.ensure_coords(coords_dict[key], dim=dim)
        centers_dict[key] = coords.mean(0)
        ccoords_dict[key] = coords - centers_dict[key]
    unknowns = _unknowns(dim)

    # pairs
    wpairs_dict = {}
    for keyA in pairs_dict:
        wpairs_dict[keyA] = {}
        for keyB in pairs_dict[keyA]:
            pairs = pairs_dict[keyA][keyB]
            if isinstance(pairs, (tuple, list)) and len(pairs) == 2:
                pairs, w = pairs
            else:
                w = np.ones(len(pairs))
            pairs = np.array(pairs, dtype=int)
            if len(pairs) > 0:
                w = assertion.ensure_numvector(w, length=pairs.shape[0])
                wpairs_dict[keyA][keyB] = nptools.recarray({
                    'A': pairs[:, 0].astype(int),
                    'B': pairs[:, 1].astype(int),
                    'weights': w.astype(float)
                })

    # try to keep the original location and orientation
    weights_dict = {}
    if weights is None:
        weights = np.ones(unknowns)
    if weights is not None:
        if isinstance(weights, dict):
            for key in weights:
                weights_dict[key] = assertion.ensure_numvector(
                    weights[key],
                    length=unknowns
                ).astype(float)
        else:
            if assertion.isnumeric(weights):
                weights = np.repeat(weights, unknowns)
            if nptools.isarray(weights):
                weights = assertion.ensure_numvector(weights, length=unknowns)
                for key in ccoords_dict.keys():
                    weights_dict[key] = weights
            else:
                m = "type '%' of 'weights' not supported" % type(weights)
                raise ValueError(m)

    return dim, ccoords_dict, centers_dict, wpairs_dict, weights_dict
