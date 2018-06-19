from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np

from . import (
    assertion,
    classification,
)
from .indexkd import IndexKD


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
    indexKD : IndexKD
        Spatial index with `n` points.
    r : positive float
        Radius to select the cluster indices of neighboured points.
        # TODO Wort Klassenzugehoerigkeit
    get_class : function
        Function to define the cluster id of a point. Recieves a list of
        cluster ids of neigboured points to define the cluster id of a point.
        Returns -1 if the point is not associated with any cluster.
    order : optional, array_like(int)
        Defines the order to apply the clustering algorithm. It can also be
        used to subsample the points to cluster. If not defined the order is
        defined by decreasing point density.
    clusters : optional, array_like(int, shape=(n))
        List of `n` integers. Each element represents the preliminary cluster
        id of a point in `indexKD`. A cluster id is a positive integer. If a
        cluster id of `-1` represents no class. If None, each element is set
        to `-1`. # TODO revise
    min_size : optional, int
        Minimum number of points associated with a cluster. If less than
        `min_size` points are associated with a cluster, the cluster is
        rejected.
    auto_set : optional, bool
        Defines weather or not cluster a id is automatically set, if -1
        (no class) was returned by `get_class`. If True, a new cluster id is
        set to `max(clusters) + 1`.

    Returns
    -------
    dict
        Dictionary of clusters. The keys correspond to the class ids. The
        values correspond to the point indices associated with the cluster.

    """
    if not isinstance(indexKD, IndexKD):
        raise ValueError("'indexKD' needs to be an instance of IndexKD")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs to be a number greater zero")

    if order is None:
        # order by density
        count = indexKD.countBall(r)
        order = np.argsort(count)[::-1]
        order = order[count[order] > 1]
    elif not (hasattr(order, '__len__') and len(order) <= len(indexKD)):
        raise ValueError("'order' needs be an array with length less or equal len(indexKD)")

    if clusters is None:
        out_clusters = -np.ones(len(indexKD), dtype=int)
    else:
        # TODO replace assert
        assert hasattr(clusters, '__len__') and len(clusters) == len(indexKD)
        out_clusters = np.array(clusters, dtype=int)
        assert len(out_clusters.shape) == 1

    if not (isinstance(min_size, int) and min_size >= 0):
        raise ValueError("'min_size' needs to be an integer equal or greater zero")
    if not isinstance(auto_set, bool):
        raise ValueError("'auto_set' needs to boolean")

    nextId = out_clusters.max() + 1
    coords = indexKD.coords()

    # calculate spatial neighbourhood
    nIdsIter = indexKD.ball(coords[order, :], r)

    for pId, nIds in zip(order, nIdsIter):
        cIds = [out_clusters[nId] for nId in nIds if out_clusters[nId] != -1]
        if len(cIds) > 0:
            out_clusters[pId] = get_class(cIds)
        elif auto_set:
            out_clusters[pId] = nextId
            nextId += 1

    return classifictation.classes_to_dict(out_clusters, min_size=min_size)


# TODO vereinfachte clustering funktionen mit vordefinierten gewichten,
# order etc.!


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



    See Also
    --------
    clustering

    """

    return clustering(
        indexKD,
        r,
        classification.mayority,
        order,
        clusters,
        min_size,
        auto_set)


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


    See Also
    --------
    clustering

    """

    if weights is None:
        weights = np.ones(len(indexKD), dtype=float)
    else:
        assert hasattr(weights, '__len__') and len(weights) == len(indexKD)
        weights = np.array(weights, dtype=float)

    def get_class(cIds):
        cWeight = defaultdict(lambda: 0)
        for cId in cIds:
            cWeight[cId] += weights[cId]
        for key in cWeight:
            if cWeight[key] > cWeight[cId]:
                cId = key
        weights[cId] = float(cWeight[cId]) / len(cIds)
        return cId

    return clustering(
        indexKD,
        r,
        get_class,
        order,
        clusters,
        min_size,
        auto_set)


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
    out_clusters = DBSCAN(eps=epsilon, min_samples=min_pts).fit_predict(coords)
    return classification.classes_to_dict(out_clusters, min_size=min_size, max_size=max_size)
