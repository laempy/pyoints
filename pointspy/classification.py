import numpy as np
from collections import defaultdict


def splitByBreaks(values, breaks):
    classes = np.digitize(values, breaks)
    for i in range(len(breaks)):
        yield np.where(classes == i)[0]


def renameDict(classDict, ids=None):
    if ids is None:
        ids = range(len(classDict))
    return dict(zip(ids, classDict.values()))


def dict2classes(classes, n, minPts=1):
    classification = -np.ones(n, dtype=int)
    for cId, ids in classes.iteritems():
        if len(ids) >= minPts:
            classification[ids] = cId
    return classification


def classes2dict(classification, ids=None, min_size=1, max_size=np.inf):
    classes = defaultdict(lambda: [])
    if ids is None:
        ids = range(len(classification))

    for id, cId in zip(ids, classification):
        classes[cId].append(id)

    if min_size > 1 or max_size < np.inf:
        for key in classes.keys():
            s = len(classes[key])
            if s < min_size or s > max_size:
                del classes[key]
    return classes


def mayority(classes):
    k = len(classes) / 2
    cCount = defaultdict(lambda: 0)
    for cId in classes:
        cCount[cId] += 1
        if cCount[cId] > k:
            return cId

    for key in cCount:
        if cCount[key] > cCount[cId]:
            cId = key

    for key in cCount:
        if cCount[key] == cCount[cId] and key != cId:
            return -1
    return cId


class Sample:

    def __init__(self, trainFraction=0.7, classes=None, groups=None):

        if groups is None:
            assert classes is not None and hasattr(classes, '__len__')

        if classes is None:
            assert groups is not None and hasattr(groups, '__len__')
            n = len(groups)
            classes = np.random.permutation(range(n)) <= n
        if groups is not None:
            groups = np.array(groups)

        self.classes = classes

        # classKeys=np.unique(classes)
        classKeys = list(set(classes))
        # print classKeys
        # groupKeys=np.unique(groups)
        groupKeys = list(set(groups))
        # print groupKeys

        ids = np.arange(len(classes))
        tIds = defaultdict(lambda: [])
        vIds = defaultdict(lambda: [])

        for classKey in classKeys:
            cIds = ids[classes == classKey]
            n = len(cIds)
            cIds = np.random.permutation(cIds)

            if groups is None:
                tMask = np.arange(n) < trainFraction * n
                vMask = ~tMask
            else:
                # avoid same group within class
                # mask=np.zeros(len(cIds),dtype=bool)
                # for groupKey in np.random.permutation(groupKeys):
                #    mask[groups[cIds]==groupKey]=True
                #    if mask.sum()>=trainFraction*n:
                #        break

                # Optimal split
                counts = {}
                for groupKey in groupKeys:
                    counts[groupKey] = (groups[cIds] == groupKey).sum()

                order = np.argsort(counts.values())[::-1]
                vMask = np.zeros(len(cIds), dtype=bool)
                tMask = np.zeros(len(cIds), dtype=bool)
                for groupKey in np.array(counts.keys())[order]:
                    mask = groups[cIds] == groupKey
                    if counts[groupKey] > 0:
                        if tMask.sum() == 0 and vMask.sum() == 0:
                            f = 0
                        else:
                            f = 1.0 * tMask.sum() / (tMask.sum() + vMask.sum())
                        if f < trainFraction:
                            tMask[mask] = True
                        else:
                            vMask[mask] = True

            tIds[classKey] = cIds[tMask]
            vIds[classKey] = cIds[vMask]

        self._tIds = tIds
        self._vIds = vIds

    @property
    def tIds(self):
        return self._tIds

    @property
    def vIds(self):
        return self._vIds

    @property
    def tMask(self):
        mask = np.zeros(len(self.classes), dtype=bool)
        for tIds in self.tIds.values():
            mask[tIds] = True
        return mask

    @property
    def vMask(self):
        mask = np.zeros(len(self.classes), dtype=bool)
        for vIds in self.vIds.values():
            mask[vIds] = True
        return mask

    def keys(self):
        return np.unique(self.classes)

    def counts(self, mask=None):
        if mask is None:
            mask = np.ones(len(self.classes), dtype=bool)
        counts = {}
        for key in self.keys():
            counts[key] = (self.classes[mask] == key).sum()
        return counts

    @property
    def tCounts(self):
        return self.counts(self.tMask)

    @property
    def vCounts(self):
        return self.counts(self.vMask)

    @property
    def masks(self):
        return self.tMask, self.vMask
