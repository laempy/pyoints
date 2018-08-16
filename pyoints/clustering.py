# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, Trier University,
# lamprecht@uni-trier.de
#
# Pyoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pyoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE
"""Clustering algorithms to assign group points.
"""

import numpy as np
from numbers import Number
from collections import defaultdict
from sklearn.cluster import DBSCAN


from . import (
    assertion,
    classification,
)
from .indexkd import IndexKD


def clustering(indexKD,
               r,
               get_class,
               order=None,
               clusters=None,
               auto_set=True):
    """Generic clustering based on spatial neighbourhood.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index with `n` points.
    r : positive float
        Radius to identify the cluster affiliation of neighboured points.
    get_class : callable
        Function to define the cluster id (affiliation) of a point. It recieves
        a list of cluster ids of neigboured points to define the cluster id of
        selected point. It returns -1 if the point is not associated with any
        cluster.
    order : optional, array_like(int)
        Defines the order to apply the clustering algorithm. It can also be
        used to subsample points for clustering. If None, the order is defined
        by decreasing point density.
    clusters : optional, array_like(int, shape=(n))
        List of `n` integers. Each element represents the preliminary cluster
        id of a point in `indexKD`. A cluster id of `-1` represents no class.
    auto_set : optional, bool
        Defines weather or not a cluster id is set automatically if -1
        (no class) was returned by `get_class`. If True, a new cluster id is
        set to `max(clusters) + 1`.

    Returns
    -------
    dict
        Dictionary of clusters. The keys correspond to the class ids. The
        values correspond to the point indices associated with the cluster.

    """
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
    if not (assertion.isnumeric(r) and r > 0):
        raise ValueError("'r' needs to be a number greater zero")
    if not callable(get_class):
        raise TypeError("'get_class' needs to be callable")

    if order is None:
        # order by density
        count = indexKD.ball_count(r)
        order = np.argsort(count)[::-1]
    else:
        order = assertion.ensure_numvector(order, max_length=len(indexKD))

    if clusters is None:
        out_clusters = -np.ones(len(indexKD), dtype=int)
    else:
        out_clusters = assertion.ensure_numvector(
            clusters,
            min_length=len(indexKD),
            max_length=len(indexKD)
        )
    if not isinstance(auto_set, bool):
        raise TypeError("'auto_set' needs to be of type boolean")

    nextId = out_clusters.max() + 1
    coords = indexKD.coords

    # calculate spatial neighbourhood
    nIdsIter = indexKD.ball_iter(coords[order, :], r)

    for pId, nIds in zip(order, nIdsIter):
        cIds = [out_clusters[nId] for nId in nIds if out_clusters[nId] != -1]
        if len(cIds) > 0:
            out_clusters[pId] = get_class(cIds)
        elif auto_set:
            out_clusters[pId] = nextId
            nextId += 1

    return out_clusters


def mayority_clusters(indexKD, r, **kwargs):
    """Clustering by mayority voting.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index with `n` points.
    r : positive float
        Radius to identify the cluster affiliation of neighboured points.
    \*\*kwargs : optional
        Optional arguments of the `clustering` function.

    See Also
    --------
    clustering

    Examples
    --------

    >>> coords = [(0, 0), (1, 1), (2, 1), (3, 3), (0, 1), (2, 3), (3, 4)]
    >>> clusters = mayority_clusters(IndexKD(coords), 2)
    >>> print(clusters)
    [ 1  1 -1  0  1  0  0]

    """
    return clustering(indexKD, r, classification.mayority, **kwargs)


def weight_clusters(indexKD, r, weights=None, **kwargs):
    """Clustering by class weight.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index with `n` points.
    r : positive float
        Radius to identify the cluster affiliation of neighboured points.
    weights : optional, array_like(Number, shape=(len(indexKD)))
        Weighting of each point. The class with highest weight wins. If None,
        all weights are set to 1, which results in similar results than
        `mayority_clusters`.
    \*\*kwargs : optional
        Optional arguments passed to `clustering`.

    Examples
    --------

    Equal weights.

    >>> coords = [(0, 0), (0, 1), (1, 1), (0, 0.5), (2, 2), (2, 2.5), (2.5, 2)]
    >>> indexKD = IndexKD(coords)
    >>> initial_clusters = np.arange(len(coords), dtype=int)

    >>> clusters = weight_clusters(indexKD, 1.5, clusters=initial_clusters)
    >>> print(clusters)
    [0 0 4 3 6 5 5]

    Differing weights.

    >>> weights = np.arange(len(coords))
    >>> clusters = weight_clusters(
    ...     indexKD,
    ...     1.5,
    ...     weights=weights,
    ...     clusters=initial_clusters
    ... )
    >>> print(clusters)
    [3 1 4 3 6 5 5]

    See Also
    --------
    clustering, mayority_clusters

    """
    if weights is None:
        weights = np.ones(len(indexKD), dtype=float)
    else:
        weights = assertion.ensure_numvector(
            weights,
            min_length=len(indexKD),
            max_length=len(indexKD)
        )

    def get_class(cIds):
        cWeight = defaultdict(lambda: 0)
        for cId in cIds:
            cWeight[cId] += weights[cId]
        for key in cWeight:
            if cWeight[key] > cWeight[cId]:
                cId = key
        weights[cId] = float(cWeight[cId]) / len(cIds)
        return cId

    return clustering(indexKD, r, get_class, **kwargs)


def dbscan(
        indexKD,
        min_pts,
        epsilon=None,
        quantile=0.8,
        factor=3):
    """Variant of the DBSCAN algorithm [1] with automatic estimation of the
    `epsilon` parameter using point density. Usefull for automatic outlier
    identification.

    Parameters
    ----------
    indexKD : IndexKD
        Spatial index with `n` points to cluster.
    min_pts : int
        Corresponds to the `min_pts` parameter of the DBSCAN algorithm.
    epsilon : optional, positive float
        Corresponds to the `epsilon` parameter of DBSCAN algorithm. If None,
        a suitable value is estimated by investigating the nearest neighbour
        distances `dists` of all points in `indexKD` with ```epsilon =
        np.percentile(dists, quantile * 100) * factor```.
    quantile : optional, positive float
        Used to calculate `epsilon`.
    factor: optional, positive float
        Used to calculate `epsilon`.

    References
    ----------

    [1] M. Ester, et al. (1996): "A Density-Based Algorithm for Discovering
    Clusters in Large Spatial Databases with Noise", KDD-96 Proceedings,
    pp. 226-231.

    Examples
    --------

    >>> coords = [(0, 0), (0, 1), (1, 1), (0, 0.5), (2, 2), (2, 2.5), (19, 29)]
    >>> indexKD = IndexKD(coords)

    User defined epsilon.

    >>> clusters = dbscan(indexKD, 1, epsilon=1)
    >>> print(clusters)
    [0 0 0 0 1 1 2]

    Automatic epsilon estimation for outlier removal.

    >>> clusters = dbscan(indexKD, 2)
    >>> print(clusters)
    [ 0  0  0  0  0  0 -1]

    Adjust automatic epsilon estimation to achieve small clusters.

    >>> clusters = dbscan(indexKD, 1, quantile=0.7, factor=1)
    >>> print(clusters)
    [0 0 1 0 2 2 3]

    """
    if not isinstance(indexKD, IndexKD):
        raise TypeError("'indexKD' needs to be of type 'IndexKD'")
    if not (isinstance(min_pts, int) and min_pts >= 0):
        m = "'min_pts' needs to be an integer greater or equal zero"
        raise ValueError(m)

    coords = indexKD.coords

    # Estimate epsilon based on density
    if epsilon is None:
        if not (isinstance(quantile, Number) and quantile > 0):
            raise ValueError("'quantile' needs to be a number greater zero")
        if not (isinstance(factor, Number) and factor > 0):
            raise ValueError("'factor' needs to be a number greater zero")

        if min_pts > 0:
            dists = indexKD.knn(coords, k=min_pts + 1)[0][:, 1:]
        else:
            dists = indexKD.nn[0]
        epsilon = np.percentile(dists, quantile * 100) * factor
    else:
        if not (isinstance(epsilon, Number) and epsilon > 0):
            raise ValueError("'epsilon' needs to be a number greater zero")

    # perform dbscan
    return DBSCAN(eps=epsilon, min_samples=min_pts).fit_predict(coords)
