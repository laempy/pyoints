import numpy as np

from . import transformation
from .IndexKD import IndexKD
from . import distance


def optRotoTranslation(coordsA, coordsB):

    # http://nghiaho.com/?page_id=671
    # http://nghiaho.com/uploads/code/rigid_transform_3D.py_
    # Zhang_2016a

    assert coordsA.shape[0] == coordsB.shape[0] and coordsA.shape[1] == coordsB.shape[1], 'dimensions do not match!'

    dim = coordsA.shape[1]

    cA = coordsA.mean(0)
    cB = coordsB.mean(0)

    A = np.matrix(transformation.homogenious(coordsA - cA, value=0))
    B = np.matrix(transformation.homogenious(coordsB - cB, value=0))

    # Find rotation matrix
    H = np.transpose(A) * B
    USV = np.linalg.svd(H)
    R = USV[0] * USV[2]

    # reflection case
    if np.linalg.det(R) < 0:
        USV[0][2, :] *= -1
        R = USV[0] * USV[2]

    # Create transformation matrix
    T1 = transformation.tMatrix(-cB)
    M = (R * T1)
    M[0:dim, dim] += np.matrix(cA).T

    return M


def optTransformation(coordsA, coordsB):

    assert coordsA.shape[0] == coordsB.shape[0] and coordsA.shape[1] == coordsB.shape[1], 'dimensions do not match!'

    b = homogenious(coordsA)
    A = homogenious(coordsB)

    x = np.linalg.lstsq(A, b)[0]
    M = np.matrix(x).T

    return M


def networkBalancing(coordsDict, pairs, sWeights=1, oWeights={}):

    # basics: http://geomatica.como.polimi.it/corsi/def_monitoring/roto-translationsb.pdf
    # Zhang_2016a
    '''
    coordsDict = { 'A': coordsA, 'B': coordsB, 'C': coordsC, 'D': coordsD }
    pairs = {
            'A': { 'B': pairsAB, 'C': pairsAC, 'D': pairsAD },
            'B': { 'A': pairsBA, 'B': pairsBC },
            'C': { 'B': pairsBC },
            'D': { 'D': pairsAD},
        }
    pairsAB = np.recarray(dtype=[('A',int),('B',int),('weights',float)])
    #pairsAB=[(idA0,idB0),(idA1,idB1),(idA2,idB2)] or
    #pairsAB=[(idA0,idB0,w0),(idA1,idB1,w1),(idA2,idB2,w2)]
    '''
    pairsDtype = [('A', int), ('B', int), ('weights', float)]

    # check dimensions
    dim = None
    for key in coordsDict:
        if dim is None:
            dim = coordsDict[key].shape[1]
        assert coordsDict[key].shape[1] == dim, 'Dimensions do not match!'

    k = len(coordsDict)  # number of point clouds

    # pairs
    wPairs = {}
    for keyA in pairs:
        wPairs[keyA] = {}
        for keyB in pairs[keyA]:
            p = np.array(pairs[keyA][keyB])
            if len(p) < 3:
                w = np.ones(len(p[0]))
            n = p[0].shape[0]
            assigned = np.recarray(n, dtype=pairsDtype)
            assigned.A = p[0]
            assigned.B = p[1]
            assigned.weights = 1

            wPairs[keyA][keyB] = assigned

    # set weights

    # sum of translations and rotations close to zero
    if hasattr(sWeights, '__len__'):
        assert len(sWeights) == 2 * dim
    else:
        sWeights = np.repeat(sWeights, 2 * dim)

    # keep original location and orientation
    if isinstance(oWeights, dict):
        for iA, keyA in enumerate(coordsDict):
            if keyA not in oWeights:
                oWeights[keyA] = np.zeros(dim * 2)
            oWeights[keyA] = np.array(oWeights[keyA], dtype=float)

    # helper function
    def subM(coords):
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

    # build linear equation system B=A*M
    A = []
    B = []
    for iA, keyA in enumerate(coordsDict):
        if keyA in pairs:

            for iB, keyB in enumerate(coordsDict):
                if keyB in pairs[keyA]:

                    # get pairs of coordinates
                    p = wPairs[keyA][keyB]
                    coordsA = coordsDict[keyA][p.A, :]
                    coordsB = coordsDict[keyB][p.B, :]

                    a = np.zeros((coordsA.shape[0] * dim, k * dim * 2))
                    a[:, iA * dim * 2:(iA + 1) * dim * 2] = subM(coordsA)
                    a[:, iB * dim * 2:(iB + 1) * dim * 2] = -subM(coordsB)

                    b = coordsB.T.flatten() - coordsA.T.flatten()

                    w = np.tile(p.weights, dim)
                    a = (a.T * w).T
                    b = b * w

                    A.append(a)
                    B.append(b)

    # try to keep the sum of translations and rotations close to zero
    a = np.tile(np.eye(dim * 2 * k, dim * 2), k)
    b = np.zeros(dim * 2 * k)

    w = np.tile(sWeights, k)
    a = (a.T * w).T
    b = b * w

    A.append(a)
    B.append(b)

    # try to keep the original locations and orientations
    for iA, keyA in enumerate(coordsDict):

        a = np.eye(dim * 2, k * dim * 2, k=iA * dim * 2)
        b = np.hstack((coordsDict[keyA].mean(0), np.zeros(dim)))

        w = oWeights[keyA]
        a = (a.T * w).T
        b = b * w

        A.append(a)
        B.append(b)

    # solve linear equation system
    A = np.vstack(A)
    B = np.hstack(B)
    M = np.linalg.lstsq(A, B)[0]

    # Extract roto-transformation matrices
    res = {}
    for iA, keyA in enumerate(coordsDict):
        T = transformation.tMatrix(M[iA * dim * 2:iA * dim * 2 + dim])
        R = transformation.rMatrix(M[iA * dim * 2 + dim:(iA + 1) * dim * 2])
        res[keyA] = T * R
    return res


def ICP(coordsDict, maxDist, k=1, p=2, maxIter=10):
    # iterative closest point

    assert k > 0
    assert isinstance(coordsDict, dict)

    dim = None
    M = {}  # Translation matrices
    for key, coords in coordsDict.items():
        if dim is None:
            dim = coords.shape[1]
        assert coords.shape[1] == dim, 'Dimensions do not match!'
        M[key] = transformation.iMatrix(dim)

    for numIter in range(maxIter):
        print numIter
        pairs = {}
        for keyA in coordsDict:
            pairs[keyA] = {}
            coordsA = transformation.transform(coordsDict[keyA], M[keyA])
            indexKD = IndexKD(coordsA)

            for keyB in coordsDict:
                if keyB != keyA:

                    coordsB = transformation.transform(
                        coordsDict[keyB], M[keyB])
                    dists, kNN = indexKD.kNN(
                        coordsB,
                        k=k,
                        distance_upper_bound=maxDist
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
                        w = distance.IDW(dists[bIds, i], p=p)
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
        M = networkBalancing(coordsDict, pairs)
        # printDists(M)

    return M
