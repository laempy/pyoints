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



def icp(coords_dict, radii, T_dict={}, assign_class=assign.KnnMatcher, max_iter=10):
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

    >>> A = np.array([(0, 0), (0, 0.1), (1, 1), (1, 0), (0.5, 0.5), (-1, -2)])
    >>> B = np.array([(0.4, 0.4), (0.2, 0), (0.1, 1.2), (2, 1), (-1.1, -1.2)])

    >>> coords_dict = {'A': A, 'B': B}
    >>> res = icp(coords_dict, (0.6, 0.6), max_iter=10)
    >>> print(res)


    """

    # prepare input

    if not isinstance(coords_dict, dict):
        raise TypeError("'coords_dict' needs to be a dictionary")
    if len(coords_dict) < 2:
        raise ValueError("at least two point sets are required")
    if not isinstance(T_dict, dict):
        raise TypeError("'T_dict' needs to be a dictionary")

    if not isinstance(assign_class, type):
        # TODO better check
        #raise TypeError("'assign_class' must be a callable object")
        pass

    radii = assertion.ensure_numvector(radii)
    #S = transformation.s_matrix(1.0 / radii)
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
                coords_dict[keyA],
                T_dict[keyA]
            )
            matcher = assign_class(coordsA, radii)

            for keyB in coords_dict:
                if keyB != keyA:

                    coordsB = transformation.transform(
                        coords_dict[keyB],
                        T_dict[keyB]
                    )

                    pairs = matcher(coordsB)
                    if len(pairs) > 0:
                        dist = distance.dist(
                            coordsA[pairs[:, 0], :],
                            coordsB[pairs[:, 1], :]
                        )
                        print dist
                        w = distance.idw(dist)
                        print w
                        pairs_dict[keyA][keyB] = [pairs[:, 0], pairs[:, 1], w]

                    # TODO: calc distances
                    # TODO: outlier removal
                        print len(pairs_dict[keyA][keyB])
        #print num_iter
        T_dict = rototranslations.find_rototranslations(coords_dict, pairs_dict)
        print T_dict[keyA]

        #print find_rototranslations(coords_dict, pairs_dict)
        #T = find_rototranslations(coords_dict, pairs_dict)
        #print T


def icp_old(coords_dict, max_dist, k=1, p=2, max_iter=10):
    """


    """
    # TODO reimplement



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
