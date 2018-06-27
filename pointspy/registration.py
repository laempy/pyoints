"""Registration or alignment of point sets.
"""

import numpy as np

from . indexkd import IndexKD
from . import (
    assertion,
    transformation,
    distance,
    nptools
)


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
    http://nghiaho.com/?page_id=671

    Examples
    --------

    >>> T = transformation.matrix(t=[3, 5], r=0.3)
    >>> A = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    >>> B = transformation.transform(A, T)
    >>> M = find_rototranslation(A, B)
    >>> C = transformation.transform(B, M, inverse=False)
    >>> print np.round(C, 2)
    [[0. 0.]
     [0. 1.]
     [1. 1.]
     [1. 0.]]

    """

    # TODO
    # http://nghiaho.com/?page_id=671
    # http://nghiaho.com/uploads/code/rigid_transform_3D.py_
    # Zhang_2016a

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

    # validate result (only if no residuals in input)
    #ixd = (0, -1)
    #close = np.isclose(transformation.transform(B[ixd, :], M), A[ixd, :])
    #assert np.all(close), "could not find an appropiate transformation matrix"

    return transformation.LocalSystem(M)


#TODO
# find_transformation with optional translation, rotation, scaling and skewing
# take a look at cv2

def find_transformation(A, B):
    """Finds the optimal (non-rigid) transformation matrix `M` between two point
    sets. Each point of point set `A` is associated with exactly one point in
    point set `B`.

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


def find_rototranslations(coords_dict, pairs_dict, weights=None):
    """Finds the optimal roto-translation matrices between multiple point sets.
    The algorithm assumes infinitissimal rotations between the point sets.

    # TODO: types
    Parameters
    ----------
    coords_dict: dict
        Dictionary of point sets with `k` dimensions.
    pairs_dict : dict
        Dictionary of point pairs.
    weights : optional, dict or list or int.
        Try to keep the original location and orientation by weighting. Each
        point set can be weighted by a list of `2 * k` values. The first `k`
        values represent the weighting factors for location. The last `k`
        values represent the weighting factors for orientation (angles).
        The weights can be provided for each point set individially in form
        of a dictionary weights. If not provided, weights of zero are assumed.

    Returns
    -------
    M : dict
        Dictionary of roto-translation matrices with `B` to `A` with
        `A = B * M.T`.

    Notes
    -----
    Currently just three dimensions are supported.
    coords_dict = { 'A': A, 'B': B, 'C': C, 'D': D }
    pairs_dict = {
            'A': { 'B': pairsAB, 'C': pairsAC, 'D': pairsAD },
            'B': { 'A': pairsBA, 'B': pairsBC },
            'C': { 'B': pairsBC },
            'D': { 'D': pairsAD },
        }
    pairsAB = np.recarray(dtype=[('A', int), ('B', int), ('weights', float)])
    or
    pairsAB=[(idA0, idB0), (idA1, idB1), (idA2, idB2)]
    or
    pairsAB=[(idA0, idB0, w0), ( idA1, idB1, w1), (idA2, idB2, w2)]

    References
    ----------
    Basic idea: http://geomatica.como.polimi.it/corsi/def_monitoring/roto-translationsb.pdf
    # Zhang_2016a


    Examples
    --------

    Prepare coordinates.

    >>> coordsA = [(-1, -2, 3), (-1, 2, 4), (1, 2, 5), (1, -2, 6)]
    >>> T = transformation.matrix(t=[10000, 20000, 3000], r=[0.01, 0.01, -0.002], order='trs')
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
    >>> print(np.round(coordsA, 2))

    """

    # TODO 2D?

    if not isinstance(coords_dict, dict):
        raise TypeError("'coords_dict' of type 'dict' required")
    k = len(coords_dict)  # number of point clouds
    if k < 2:
        raise ValueError("at least 2 point sets required")

    # check dimensions
    dim = None
    for key in coords_dict:
        coords_dict[key] = assertion.ensure_coords(
            coords_dict[key],
            min_dim=2,
            max_dim=3
        )
        if dim is None:
            dim = coords_dict[key].shape[1]
        assert coords_dict[key].shape[1] == dim, 'Dimensions do not match!'

    if not dim == 3:
        print(dim)
        raise ValueError("%i dimensions are not supported yet" % dim)

    # pairs
    wPairs = {}
    dtype_pairs = [('A', int), ('B', int), ('weights', float)]
    for keyA in pairs_dict:
        wPairs[keyA] = {}
        for keyB in pairs_dict[keyA]:
            p = np.array(pairs_dict[keyA][keyB])
            if p.shape[1] < 3:
                w = np.ones(len(p))
            n = p.shape[0]
            assigned = np.recarray(n, dtype=dtype_pairs)

            assigned.A = p[:, 0]
            assigned.B = p[:, 1]
            assigned.weights = w
            wPairs[keyA][keyB] = assigned

    # try to keep the original location and orientation
    oWeights = {}
    if weights is not None:
        if isinstance(weights, dict):
            for key in weights:
                oWeights[key] = assertion.ensure_numvector(
                    weights[key],
                    length=2 * dim
                ).astype(float)
        else:
            if nptools.isarray(weights):
                weights = assertion.ensure_numvector(weights, length=2 * dim)
            else:
                m = "type '%' of 'weights' not supported" % type(weights)
                raise ValueError(m)
            for key in coords_dict.keys():
                oWeights[key] = weights

    # helper function
    # TODO n-dimensional
    def get_equations(coords):
        N, dim = coords.shape

        Mx = np.zeros((N, 2 * dim))
        Mx[:, 0] = 1  # t_x
        Mx[:, 4] = coords[:, 2]  # z
        Mx[:, 5] = -coords[:, 1]  # -y

        My = np.zeros((N, 2 * dim))
        My[:, 1] = 1  # t_y
        My[:, 3] = -coords[:, 2]  # -z
        My[:, 5] = coords[:, 0]  # x

        Mz = np.zeros((N, 2 * dim))
        Mz[:, 2] = 1  # t_z
        Mz[:, 3] = coords[:, 1]  # y
        Mz[:, 4] = -coords[:, 0]  # -x

        return np.vstack((Mx, My, Mz))


    centers = {}
    for key in coords_dict:
        centers[key] = coords_dict[key].mean(0)

    # build linear equation system mA = mB * M

    mA = []
    mB = []
    for iA, keyA in enumerate(coords_dict):
        if keyA in pairs_dict:
            for iB, keyB in enumerate(coords_dict):
                if keyB in pairs_dict[keyA]:

                    # get pairs of points
                    p = wPairs[keyA][keyB]
                    A = coords_dict[keyA][p.A, :] - centers[keyA]
                    B = coords_dict[keyB][p.B, :] - centers[keyB]

                    # set equations
                    a = np.zeros((A.shape[0] * dim, k * dim * 2))
                    a[:, iA * dim * 2:(iA + 1) * dim * 2] = get_equations(A)
                    a[:, iB * dim * 2:(iB + 1) * dim * 2] = -get_equations(B)

                    b = B.T.flatten() - A.T.flatten()

                    # weighting
                    w = np.tile(p.weights, dim)
                    a = (a.T * w).T
                    b = b * w

                    mA.append(a)
                    mB.append(b)

    # try to keep the original locations and orientations
    for i, key in enumerate(coords_dict):
        if key in oWeights:

            a = np.eye(dim * 2, k * dim * 2, k=i * dim * 2)
            b = np.zeros(2 * dim)

            a = np.eye(dim * 2, k * dim * 2, k=i * dim * 2)

            b = np.hstack([centers[key], np.zeros(dim)])
            #b = np.hstack([np.ones(dim), np.zeros(dim)])
            #b = np.hstack([np.zeros(dim), np.zeros(dim)])

            w = oWeights[key]
            a = (a.T * w).T
            b = b * w
            print(a)
            print(b)

            mA.append(a)
            mB.append(b)

    # solve linear equation system
    mA = np.vstack(mA)
    mB = np.hstack(mB)
    M = np.linalg.lstsq(mA, mB, rcond=None)[0]

    # Extract roto-transformation matrices
    res = {}
    for iA, keyA in enumerate(coords_dict):
        t = M[iA * dim * 2:iA * dim * 2 + dim]
        r = M[iA * dim * 2 + dim:(iA + 1) * dim * 2]
        #print(t)
        #T = transformation.t_matrix(t + centers[keyA])
        #R = transformation.r_matrix(r)
        #res[keyA] = T * R   # do not edit!!!
        #res[keyA] = TC * R

        T1 = transformation.t_matrix(-centers[keyA])
        T2 = transformation.t_matrix(t)
        R = transformation.r_matrix(r)
        res[keyA] = T2 * R * T1

    return res


def ICP(coords_dict, max_dist, k=1, p=2, max_iter=10):
    # TODO docu

    # iterative closest point

    assert k > 0
    assert isinstance(coords_dict, dict)

    dim = None
    M = {}  # Translation matrices
    for key, coords in coords_dict.items():
        if dim is None:
            dim = coords.shape[1]
        assert coords.shape[1] == dim, 'Dimensions do not match!'
        M[key] = transformation.i_matrix(dim)

    for numIter in range(max_iter):
        pairs = {}
        for keyA in coords_dict:
            pairs[keyA] = {}
            coordsA = transformation.transform(coords_dict[keyA], M[keyA])
            indexKD = IndexKD(coordsA)

            for keyB in coords_dict:
                if keyB != keyA:

                    coordsB = transformation.transform(
                        coords_dict[keyB], M[keyB])
                    dists, kNN = indexKD.knn(
                        coordsB,
                        k=k,
                        distance_upper_bound=max_dist
                    )

                    if k == 1:
                        kNN = np.array([kNN]).T
                        dists = np.array([dists]).T

                    # assign pairs
                    aIdsList = []
                    bIdsList = []
                    wList = []
                    for i in range(k):
                        bIds = np.where(dists[:, i] < np.inf)[0]
                        aIds = kNN[:, i][bIds]
                        w = distance.idw(dists[bIds, i], p=p)
                        aIdsList.extend(aIds)
                        bIdsList.extend(bIds)
                        wList.extend(w)

                    pairs[keyA][keyB] = (aIdsList, bIdsList, wList)

        # def printDists(M):
        #    print
        #    print 'Dists'
        #    for keyA in coords_dict:
        #        coordsA = transformation.transform(coords_dict[keyA],M[keyA])
        #        for keyB in coords_dict:
        #            if keyB != keyA:
        #                coordsB = transformation.transform(coords_dict[keyB],M[keyB])
        #                print keyA,keyB
        #                assigned=pairs[keyA][keyB]
        #                print distance.rmse(coordsA[assigned[0],:],coordsB[assigned[1],:])

        #print '-----'
        # printDists(M)
        M = find_rototranslations(coords_dict, pairs)
        # printDists(M)

    return M
