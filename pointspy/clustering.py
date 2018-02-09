import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

from .classification import mayority, classes2dict


def dbscan(
        indexKD,
        epsilon=None,
        quantile=0.9,
        quantileFactor=3,
        minPts=1,
        minSize=1,
        maxSize=np.inf):

    coords = indexKD.coords()

    if epsilon is None:
        # dists=indexKD.kNN(coords,minPts+1)[0][:,1:]
        if minPts > 0:
            dists = indexKD.kNN(coords, k=minPts + 1)[0][:, 1:]
        else:
            dists = indexKD.NN()[0]
        epsilon = np.percentile(dists, quantile * 100) * quantileFactor

    classification = DBSCAN(
        eps=epsilon,
        min_samples=minPts).fit_predict(coords)
    clusters = classes2dict(classification, minSize=minSize, maxSize=maxSize)
    return clusters


def mayorityClustering(
        indexKD,
        r,
        order,
        classes=None,
        minPts=1,
        autoSet=True):
    coords = indexKD.coords()

    classification = -np.ones(len(indexKD),
                              dtype=int) if classes is None else np.copy(classes)
    nextId = classification.max() + 1

    for pId in order:
        nIds = indexKD.ball(coords[pId, :], r)
        cIds = [classification[nId]
                for nId in nIds if classification[nId] != -1]
        if len(cIds) > 0:
            classification[pId] = mayority(cIds)
        elif autoSet:
            classification[pId] = nextId
            nextId += 1

    return classes2dict(classification, minSize=minPts)


def weightClustering(
        indexKD,
        r,
        order,
        weights=None,
        classes=None,
        minPts=1,
        autoSet=True):
    coords = indexKD.coords()

    classification = -np.ones(len(indexKD),
                              dtype=int) if classes is None else np.copy(classes)
    weights = np.ones(
        len(indexKD),
        dtype=float) if weights is None else np.copy(weights)
    nextId = classification.max() + 1

    def getClass(cIds):
        cWeight = defaultdict(lambda: 0)
        for cId in cIds:
            # if weights[cId]<1:
            #    print weights[cId]
            cWeight[cId] += weights[cId]

        for key in cWeight:
            if cWeight[key] > cWeight[cId]:
                cId = key

        # for key in cWeight:
        #    if cWeight[key]==cWeight[cId] and key!=cId:
        #        return -1,1
        return cId, float(cWeight[cId]) / len(cIds)

    for pId in order:
        nIds = indexKD.ball(coords[pId, :], r)
        cIds = [classification[nId]
                for nId in nIds if classification[nId] != -1]
        if len(cIds) > 0:
            cId, w = getClass(cIds)
            classification[pId] = cId
            weights[pId] = w
        elif autoSet:
            classification[pId] = nextId
            nextId += 1

    return classes2dict(classification, minSize=minPts)


def densityClustering(indexKD, r, classes=None):

    coords = indexKD.coords()

    count = indexKD.countBall(1.0 * r)
    order = np.argsort(count)[::-1]
    order = order[count[order] > 1]

    outClasses = -np.ones(len(indexKD),
                          dtype=int) if classes is None else np.copy(classes)

    for pId in order:
        nIds = indexKD.ball(coords[pId, :], r)
        cIds = [outClasses[nId] for nId in nIds if outClasses[nId] != -1]
        outClasses[pId] = pId if len(cIds) == 0 else mayority(cIds)

    return classification
