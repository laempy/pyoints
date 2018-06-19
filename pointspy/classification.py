import numpy as np
from collections import defaultdict

from . import assertion


def classes_to_dict(classification, ids=None, min_size=1, max_size=np.inf):
    """Converts a list of class indices to an dictionary of grouped classes.

    Parameters
    ----------

    Returns
    -------


    See Also
    --------
    dict_to_classes

    Examples
    --------

    >>> classes = [0, 0, 1, 2, 1, 0, 3, 3, 5, 3, 2, 1, 0]
    >>> print(classes_to_dict(classes))
    {0: [0, 1, 5, 12], 1: [2, 4, 11], 2: [3, 10], 3: [6, 7, 9], 5: [8]}

    """
    if ids is None:
        ids = range(len(classification))

    # initialize
    classes = {}
    for cId in classification:
        classes[cId] = []

    # set values
    for id, cId in zip(ids, classification):
        classes[cId].append(id)

    # check size
    if min_size > 1 or max_size < np.inf:
        for key in classes.keys():
            s = len(classes[key])
            if s < min_size or s > max_size:
                del classes[key]

    return classes


def dict_to_classes(classes_dict, n, min_pts=1):
    """Converts a dictionary of classes to a list of classes.

    Parameters
    ----------
    classes_dict : dict
        Dictionary of classes.
    n : positive int
        Size of output array.
    min_pts : int
        Minimum size of a class to be kept.

    Returns
    -------
    np.ndarray(int, shape=(n))

    See Also
    --------
    classes_to_dict

    Examples
    --------

    >>> classes_dict = {0: [0, 1, 5], 1: [3, 6], 2: [7, 2]}
    >>> print(dict_to_classes(classes_dict, 10))

    """
    # TODO validation

    classification = -np.ones(n, dtype=int)
    for cId, ids in classes_dict.iteritems():
        if len(ids) >= min_pts:
            classification[ids] = cId
    return classification


def split_by_breaks(values, breaks):
    """Assign classes to values using specific value ranges.

    Parameters
    ----------
    values : array_like(Number, shape=(n))
        Values to classify.
    breaks : array_like(Number, shape=(m))
        Series of value ranges.

    Returns
    -------
    dict
        Dictionary of length `m` + 1. Each key corresponds to a class id. The
        dictionary values correspond to the value indices.

    Examples
    --------

    >>> values = np.arange(10)
    >>> breaks = [0.5, 5, 7.5]
    >>> classes = split_by_breaks(values, breaks)
    >>> print(classes)
    [0 1 1 1 1 2 2 2 3 3]

    """
    values = assertion.ensure_numvector(values)
    breaks = assertion.ensure_numvector(breaks)

    return np.digitize(values, breaks)



def rename_dict(class_dict, ids=None):
    if ids is None:
        ids = range(len(class_dict))
    return dict(zip(ids, class_dict.values()))



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


# TODO nur zur Klassifikation benoetigt ==> delete?
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
