"""Registration or alignment of point sets.
"""

import numpy as np

from . indexkd import IndexKD
from . import transformation
from . import distance
from . import assertion


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

    >>> T = transformation.matrix(t=[3, 5], r=0.8)
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

    # validate result
    ixd = (0, -1)
    close = np.isclose(transformation.transform(B[ixd, :], M), A[ixd, :])
    assert np.all(close), "could not find an appropiate transformation matrix"

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


def find_rototranslations(coordsDict, pairs, sWeights=1, oWeights={}):
    """Finds the optimal roto-translation matrices between multiple point sets.
    The algorithm assumes infinitissimal rotations between the point sets.

    # TODO: types
    Parameters
    ----------
    coordsDict: dict
        Dictionary of point sets.
    pairs : dict
        Dictionary of point pairs
    sWeights : optional, TODO
        Weights for sum of translations and rotations
    oWeights : optional, TODO
        Weights try to keep the original location and orientation TOOD

    Returns
    -------
    M : dict
        Dictionary of roto-translation matrices with `B` to `A` with A = `B * M.T`.
    """

    # basics: http://geomatica.como.polimi.it/corsi/def_monitoring/roto-translationsb.pdf
    # Zhang_2016a
    '''
    coordsDict = { 'A': A, 'B': B, 'C': C, 'D': D }
    pairs = {
            'A': { 'B': pairsAB, 'C': pairsAC, 'D': pairsAD },
            'B': { 'A': pairsBA, 'B': pairsBC },
            'C': { 'B': pairsBC },
            'D': { 'D': pairsAD },
        }
    pairsAB = np.recarray(dtype=[('A', int), ('B', int), ('weights', float)])
    #pairsAB=[(idA0, idB0), (idA1, idB1), (idA2, idB2)] or
    #pairsAB=[(idA0, idB0, w0),( idA1, idB1, w1),(idA2, idB2, w2)]
    '''

    # TODO asserts
    # TODO 2D?

    k = len(coordsDict)  # number of point clouds

    # check dimensions
    dim = None
    for key in coordsDict:
        if dim is None:
            dim = coordsDict[key].shape[1]
        assert coordsDict[key].shape[1] == dim, 'Dimensions do not match!'

    # pairs
    wPairs = {}
    for keyA in pairs:
        wPairs[keyA] = {}
        for keyB in pairs[keyA]:
            p = np.array(pairs[keyA][keyB])
            if len(p) < 3:
                w = np.ones(len(p[0]))
            n = p[0].shape[0]
            assigned = np.recarray(
                n, dtype=[('A', int),
                          ('B', int),
                          ('weights', float)])
            assigned.A = p[0]
            assigned.B = p[1]
            assigned.weights = 1

            wPairs[keyA][keyB] = assigned

    # try to keep the sum of translations and rotations close to zero
    if hasattr(sWeights, '__len__'):
        assert len(sWeights) == 2 * dim
    else:
        sWeights = np.repeat(sWeights, 2 * dim)

    # try to keep the original location and orientation
    if isinstance(oWeights, dict):
        for iA, keyA in enumerate(coordsDict):
            if keyA not in oWeights:
                oWeights[keyA] = np.zeros(dim * 2)
            oWeights[keyA] = np.array(oWeights[keyA], dtype=float)

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

    # build linear equation system mA = mB * M
    mA = []
    mB = []
    for iA, keyA in enumerate(coordsDict):
        if keyA in pairs:

            for iB, keyB in enumerate(coordsDict):
                if keyB in pairs[keyA]:

                    # get pairs of points
                    p = wPairs[keyA][keyB]
                    A = coordsDict[keyA][p.A, :]
                    B = coordsDict[keyB][p.B, :]

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

    # try to keep the sum of translations and rotations close to zero
    a = np.tile(np.eye(dim * 2 * k, dim * 2), k)
    b = np.zeros(dim * 2 * k)

    w = np.tile(sWeights, k)
    a = (a.T * w).T
    b = b * w

    mA.append(a)
    mB.append(b)

    # try to keep the original locations and orientations
    for iA, keyA in enumerate(coordsDict):

        a = np.eye(dim * 2, k * dim * 2, k=iA * dim * 2)
        b = np.hstack((coordsDict[keyA].mean(0), np.zeros(dim)))

        w = oWeights[keyA]
        a = (a.T * w).T
        b = b * w

        mA.append(a)
        mB.append(b)

    # solve linear equation system
    mA = np.vstack(mA)
    mB = np.hstack(mB)
    M = np.linalg.lstsq(mA, mB, rcond=None)[0]

    # Extract roto-transformation matrices
    res = {}
    for iA, keyA in enumerate(coordsDict):
        T = transformation.t_matrix(M[iA * dim * 2:iA * dim * 2 + dim])
        R = transformation.r_matrix(M[iA * dim * 2 + dim:(iA + 1) * dim * 2])
        res[keyA] = T * R
    return res


def ICP(coordsDict, max_dist, k=1, p=2, max_iter=10):
    # TODO docu

    # iterative closest point

    assert k > 0
    assert isinstance(coordsDict, dict)

    dim = None
    M = {}  # Translation matrices
    for key, coords in coordsDict.items():
        if dim is None:
            dim = coords.shape[1]
        assert coords.shape[1] == dim, 'Dimensions do not match!'
        M[key] = transformation.i_matrix(dim)

    for numIter in range(max_iter):
        pairs = {}
        for keyA in coordsDict:
            pairs[keyA] = {}
            coordsA = transformation.transform(coordsDict[keyA], M[keyA])
            indexKD = IndexKD(coordsA)

            for keyB in coordsDict:
                if keyB != keyA:

                    coordsB = transformation.transform(
                        coordsDict[keyB], M[keyB])
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
        #    for keyA in coordsDict:
        #        coordsA = transformation.transform(coordsDict[keyA],M[keyA])
        #        for keyB in coordsDict:
        #            if keyB != keyA:
        #                coordsB = transformation.transform(coordsDict[keyB],M[keyB])
        #                print keyA,keyB
        #                assigned=pairs[keyA][keyB]
        #                print distance.rmse(coordsA[assigned[0],:],coordsB[assigned[1],:])

        #print '-----'
        # printDists(M)
        M = find_rototranslations(coordsDict, pairs)
        # printDists(M)

    return M
