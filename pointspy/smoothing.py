import numpy as np

from IndexKD import IndexKD



def mean_ball(coords, r, numIter=1, updatePairs=False):
    """Smoothing of spatial structures by averaging neighboured point 
    coordinates.

    Parameters
    ----------
    coords: (n,k), `numpy.ndarray`
        Array representing n points with k dimensions.
    r: `float`
        Maximum distance to nearby points which are used to calculate the
        coordinate average.
    numIter: optional, `int`
        Number of iterations.
    updatePairs: optional, `bool`
        Specifies weather or not point pairs are updated on each iteration.
    """
    assert isinstance(coords,np.ndarray)
    assert isinstance(r,float) or isinstance(r,int)
    assert isinstance(numIter,int) and numIter>0
    assert isinstance(updatePairs,bool)
    
    ids = None
    mCoords = np.copy(coords)
    for _ in range(numIter):

        if ids is None or updatePairs:
            indexKD = IndexKD(mCoords)
            ids = indexKD.ball(indexKD.coords(), r)

        # averaging
        mCoords = np.array([mCoords[nIds, :].mean(0) for nIds in ids])

    return mCoords


def mean_knn(coords, k, numIter=1, updatePairs=False):
    """Smoothing of spatial structures by averaging neighboured point 
    coordinates.

    Parameters
    ----------
    coords: (n,k), `numpy.ndarray`
        Array representing n points with k dimensions.
    k: `float`
        Number of nearest points which are used to calculate the coordinate 
        average.
    numIter: optional, `int`
        Number of iterations.
    updatePairs: optional, `bool`
        Specifies weather or not point pairs are updated on each iteration.
    """
    
    assert isinstance(coords,np.ndarray)
    assert isinstance(k,int) and k>0
    assert isinstance(numIter,int) and numIter>0
    assert isinstance(updatePairs,bool)
    
    ids = None
    mCoords = np.copy(coords)
    for _ in range(numIter):

        if ids is None or updatePairs:
            indexKD = IndexKD(mCoords)
            ids = indexKD.kNN(indexKD.coords(), k=k)[1]

        # averaging
        mCoords = np.array([mCoords[nIds, :].mean(0) for nIds in ids])
        
    return mCoords
