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
        clusters=None,
        min_size=1,
        auto_set=True):
    """Generic clustering based on spatial neighbourhood.

    Parameters
    ----------
    indexKD: `IndexKD`
        Spatial index with `n` points.
    r: positive, `float`
        Radius to select the cluster ids of neighboured points. # TODO Wort Klassenzugehoerigkeit
    get_class: `function`
        Function to define the cluster id of a point. Recieves a list of
        cluster ids of neigboured points to define the cluster id of a point. 
        Returns -1 if the point is not associated with any cluster.
    order: optional, `array_like`
        TODO
        If not defined the order is defined by decreasing point density.
    clusters: optional, `array_like of ints` # TODO wie ausdruecken
        List of `n` integers. Each element represents the preliminary cluster id
        of a point in `indexKD`. A cluster id is a positive integer. If a 
        cluster id of `-1` represents no class. If None, each element is set 
        to `-1`. # TODO revise
    min_size: optional, `int`
        Minimum number of points associated with a cluster. If less than 
        `min_size` points are associated with a cluster, the cluster is rejected.
    auto_set: optiona, `boolean`
        Defines weather or not cluster a id is automatically set, if -1 (no class)
        was returned by `get_class`. If True, a new cluster id is set to 
        `max(clusters) + 1`.

    Returns
    -------
    clusters: `dict`
        Dictionary of clusters. The keys correspond to the class ids. The values
        correspond to the point indices associated with the cluster.
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

    if clusters is None:
        outclusters = -np.ones(len(indexKD),dtype=int)
    else:
        assert hasattr(clusters,'__len__') and len(clusters) == len(indexKD)
        outclusters = np.array(clusters,dtype=int)
        assert len(outclusters.shape) == 1
    
    assert isianstance(min_size,int) and min_size >= 0
    assert isianstance(auto_set,bool)    
            
    nextId = outclusters.max() + 1
    coords = indexKD.coords()
    
    # calculate spatial neighbourhood
    nIdsIter = indexKD.ball(coords[order, :], r)
    
    for pId, nIds in zip(order,nIdsIter):
        cIds = [outclusters[nId] for nId in nIds if outclusters[nId] != -1]
        if len(cIds) > 0:
            outclusters[pId] = get_class(cIds)
        elif auto_set:
            outclusters[pId] = nextId
            nextId += 1

    return classes2dict(outclusters, min_size=min_size)


                

# TODO vereinfachte clustering funktionen mit vordefinierten gewichten, order etc.!



def mayorityclusters(
        indexKD,
        r,
        order=None,
        clusters=None,
        min_size=1,
        auto_set=True):
    """Clustering by mayority voting.

    # TODO erben von Funktion?

    Parameters
    ----------
   
    """
            
    return clustering(indexKD,r,mayority,order,clusters,min_size,auto_set)


def weightclusters(
        indexKD,
        r,
        order,
        weights=None,
        clusters=None,
        min_size=1,
        auto_set=True):
    # TODO doku
    """Clustering by class weight.
    
    Parameters
    ----------
   
    """
            
    if weights is None:
        weights = np.ones(len(indexKD),dtype=float)
    else:
        assert hasattr(weights,'__len__') and len(weights) == len(indexKD)
        weights = np.array(weights,dtype=float)

    def get_class(cIds):
        cWeight = defaultdict(lambda: 0)
        for cId in cIds:
            cWeight[cId] += weights[cId]
        for key in cWeight:
            if cWeight[key] > cWeight[cId]:
                cId = key
        weights[pId] = float(cWeight[cId]) / len(cIds)
        return cId
    
    return clustering(indexKD,r,get_class,order,clusters,min_size,auto_set)
    
    
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
    outclusters = DBSCAN(eps=epsilon,min_samples=min_pts).fit_predict(coords)
    return classes2dict(outclusters, min_size=min_size, max_size=max_size)