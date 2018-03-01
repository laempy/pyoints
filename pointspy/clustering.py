import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

from .classification import (
    mayority,
    classes2dict,
)

def clustering(indexKD,
        r,
        get_class,
        order,
        classes=None,
        min_size=1,
        auto_set=True):
    """Generic clustering.

    Parameters
    ----------
    indexKD: `IndexKD`
        Spatial index with `n` points.
    r: positive, `float`
        Radius to select the neighboured points.
    get_class: `function`
        Function recieves a list of class ids and returns a class id. TODO
    order: optional, `array_like`
        TODO
        If not defined the order is defined by decreasing point density.
    classes: optional, `array_like`
        Array of `n` preliminary class ids. A class id of `-1` is accociated
        with no class.
    min_size: `int`
        TODO
    auto_set: `boolean`
        TODO

    Returns
    -------
    classification: `np.ndarray`
        Array of `n` class ids.
    """
    
    assert isianstance(indexKD,IndexKD)
    assert (isianstance(r,float) or isianstance(r,int)) and r > 0

    if order is None:
        # order by density
        count = indexKD.countBall(r)
        order = np.argsort(count)[::-1]
        order = order[count[order] > 1]
    else:
        assert hasattr(order,'__len__') and len(order) <= len(indexKD)

    if classes is None:
        outclasses = -np.ones(len(indexKD),dtype=int)
    else:
        assert hasattr(classes,'__len__') and len(classes) == len(indexKD)
        outclasses = np.array(classes,dtype=int)
        assert len(outclasses.shape) == 1
    
    assert isianstance(min_size,int) and min_size >= 0
    assert isianstance(auto_set,bool)    
            
    nextId = outclasses.max() + 1
    coords = indexKD.coords()
    
    nIdsIter = indexKD.ball(coords[order, :], r)
    
    for pId, nIds in zip(order,nIdsIter):
        cIds = [outclasses[nId] for nId in nIds if outclasses[nId] != -1]
        if len(cIds) > 0:
            outclasses[pId] = get_class(cIds)
        elif auto_set:
            outclasses[pId] = nextId
            nextId += 1

    return classes2dict(outclasses, min_size=min_size)


                

# TODO vereinfachte clustering funktionen mit vordefinierten gewichten, order etc.!



def mayorityclusters(
        indexKD,
        r,
        order=None,
        classes=None,
        min_size=1,
        auto_set=True):
    # TODO doku
            
    return clustering(indexKD,r,mayority,order,classes,min_size,auto_set)


def weightclusters(
        indexKD,
        r,
        order,
        weights=None,
        classes=None,
        min_size=1,
        auto_set=True):
    # TODO doku
            
    if weights is None:
        weights = np.ones(len(indexKD),dtype=float)
    else:
        weights = np.copy(weights)

    def get_class(cIds):
        cWeight = defaultdict(lambda: 0)
        for cId in cIds:
            cWeight[cId] += weights[cId]
        for key in cWeight:
            if cWeight[key] > cWeight[cId]:
                cId = key
        weights[pId] = float(cWeight[cId]) / len(cIds)
        return cId
    
    return clustering(indexKD,r,get_class,order,classes,min_size,auto_set)
    
    
def dbscan(
        indexKD,
        epsilon=None,
        quantile=0.9,
        factor=3,
        min_pts=1,
        min_size=1,
        max_size=np.inf):
    # TODO doku

    coords = indexKD.coords()
    
    if epsilon is None:
        # Estimate epsilon based on density
        if min_pts > 0:
            dists = indexKD.kNN(coords, k=min_pts + 1)[0][:, 1:]
        else:
            dists = indexKD.NN()[0]
        epsilon = np.percentile(dists, quantile * 100) * factor

    # perform dbscan
    outclasses = DBSCAN(eps=epsilon,min_samples=min_pts).fit_predict(coords)
    return classes2dict(outclasses, min_size=min_size, max_size=max_size)