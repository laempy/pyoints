import numpy as np

import matrix
from IndexKD import IndexKD
import npTools


def meanBall(coords, r, numIter=1, updatePairs=False):

    ids = None
    mCoords = np.copy(coords)
    for _ in range(numIter):

        if ids is None or updatePairs:
            indexKD = IndexKD(mCoords)
            ids = indexKD.ball(indexKD.coords(), r)

        mCoords = np.array([mCoords[nIds, :].mean(0) for nIds in ids])

    return mCoords


def meanKnn(coords, k, T=None, numIter=1, updatePairs=False):
    ids = None
    mCoords = np.copy(coords)
    for _ in range(numIter):

        if ids is None or updatePairs:
            indexKD = IndexKD(mCoords)
            ids = indexKD.kNN(indexKD.coords(), k=k)[1]

        mCoords = np.array([mCoords[nIds, :].mean(0) for nIds in ids])
    return mCoords
