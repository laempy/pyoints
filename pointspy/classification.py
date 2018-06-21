"""Collection of fucntions which help to classify, reclassify or clustering.
"""

import numpy as np
from collections import defaultdict

from . import (
    assertion,
    nptools,
)


def classes_to_dict(
        classification,
        ids=None,
        min_size=1,
        max_size=np.inf,
        missing_value=-1):
    """Converts a list of class indices to an dictionary of grouped classes.

    Parameters
    ----------
    classification : array_like(shape=(n))
        Array of classes
    ids : optional, array_like(int, shape=(n))
        Desired indices or IDs of classes in the output. If None, the indices
        are numbered consecutively.
    min_size : optional, positive int
        Minimum desired size of a class to be kept in the result.
    max_size : optional, positive int
        Maximum desired size of a class to be kept in the result.
    missing_value : optional, object
        Default value for missing class.

    Returns
    -------
    dict
        Dictionary of class indices. The dictionary keys represent the class
        ids, while the values represent the indices in the original array.

    See Also
    --------
    dict_to_classes

    Examples
    --------

    >>> classes = ['cat', 'cat', 'dog', 'bird', 'dog', 'bird', 'cat', 'dog']
    >>> print(classes_to_dict(classes))
    {'cat': [0, 1, 6], 'bird': [3, 5], 'dog': [2, 4, 7]}

    >>> classes = [0, 0, 1, 2, 1, 0, 3, 3, 5, 3, 2, 1, 0]
    >>> print(classes_to_dict(classes))
    {0: [0, 1, 5, 12], 1: [2, 4, 11], 2: [3, 10], 3: [6, 7, 9], 5: [8]}

    """
    if not nptools.isarray(classification):
        raise ValueError("'classification' needs to an array like object")

    if ids is None:
        ids = range(len(classification))
    elif not len(ids) == len(classification):
        m = "'classification' and 'ids' must have the same length"
        raise ValueError(m)

    # set values
    classes = defaultdict(list)
    for id, cId in zip(ids, classification):
        if not cId == missing_value:
            classes[cId].append(id)

    # check size
    if min_size > 1 or max_size < np.inf:
        for key in list(classes.keys()):
            s = len(classes[key])
            if s < min_size or s > max_size:
                del classes[key]

    return dict(classes)


def dict_to_classes(
        classes_dict,
        n,
        min_size=1,
        max_size=np.inf,
        missing_value=-1):
    """Converts a dictionary of classes to a list of classes.

    Parameters
    ----------
    classes_dict : dict
        Dictionary of class indices.
    n : positive int
        Desired size of the output array. It must be at least the size of the
        maximum class index.
    min_pts : optional, int
        Minimum size of a class to be kept in the result.
    min_size : optional, positive int
        Minimum desired size of a class to be kept in the result.
    max_size : optional, positive int
        Maximum desired size of a class to be kept in the result.
    missing_value : optional, object
        Default value for missing class.

    Returns
    -------
    np.ndarray(int, shape=(n))

    See Also
    --------
    classes_to_dict

    Notes
    -----
    Only a limited input validation is provided.

    Examples
    --------

    Alphanumeric classes.

    >>> classes_dict = {'bird': [0, 1, 5, 4], 'dog': [3, 6, 8], 'cat': [7]}
    >>> print(dict_to_classes(classes_dict, 10, missing_value=''))
    ['bird' 'bird' '' 'dog' 'bird' 'bird' 'dog' 'cat' 'dog' '']

    Omit small classes.

    >>> print(dict_to_classes(classes_dict, 10, min_size=2))
    ['bird' 'bird' '-1' 'dog' 'bird' 'bird' 'dog' '-1' 'dog' '-1']


    Numeric classes.

    >>> classes_dict = {0: [0, 1, 5], 1: [3, 6], 2: [7, 2]}
    >>> print(dict_to_classes(classes_dict, 9))
    [ 0  0  2  1 -1  0  1  2 -1]

    """
    # type validation
    if not isinstance(classes_dict, dict):
        raise TypeError("dictionary required")
    if not isinstance(n, int) and n > 0:
        raise ValueError("'n' needs to be an integer greater zero")

    # prepare output
    dt = np.array(classes_dict.keys()).dtype
    classification = np.full(n, missing_value, dtype=dt)

    # assign classes
    for cId, ids in classes_dict.iteritems():
        if len(ids) >= min_size and len(ids) <= max_size:
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


def rename_dict(d, ids=None):
    """Assigns new key names to a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to rename
    ids : optional, array_like(shape=(len(d)))
        Desired key names. If None the keys are numbered consecutively.

    Returns
    -------
    dict
        Dictionary with new names.

    Examples
    --------

    >>> d = {1: [0, 1], 2: None, 3: 'text'}
    >>> print(rename_dict(d, ['first', 'second', 'last']))
    {'second': None, 'last': 'text', 'first': [0, 1]}

    """

    if not isinstance(d, dict):
        raise TypeError("dictionary required")
    if ids is None:
        ids = range(len(d))
    elif not len(ids) == len(d):
        raise ValueError("same number of keys required")

    return dict(zip(ids, d.values()))


def mayority(classes, empty_value=-1):
    """Find most frequent class or value in an array.

    Parameters
    ----------
    classes : array_like(object, shape=(n))
        Classes or values to check.
    empty_value : object
        Class value in case that no decision can be made.

    Returns
    -------
    object
        Most frequent class.

    Notes
    -----
    Only a limited input validation is provided.

    Examples
    --------

    Find mayority class.

    >>> classes =['cat', 'dog', 'dog', 'bird', 'cat', 'dog']
    >>> print(mayority(classes))
    dog

    >>> classes =[1, 8, 9, 0, 0, 2, 4, 2, 4, 3, 2, 3, 5, 6]
    >>> print(mayority(classes))
    2

    No decision possible.

    >>> classes =[1, 2, 3, 4, 4, 3]
    >>> print(mayority(classes))
    -1

    """
    if not nptools.isarray(classes):
        raise ValueError("'classes' needs to be an array like object")

    k = len(classes) // 2
    count = defaultdict(lambda: 0)
    for cId in classes:
        count[cId] += 1
        if count[cId] > k:
            return cId

    for key in count:
        if count[key] > count[cId]:
            cId = key

    for key in count:
        if count[key] == count[cId] and key != cId:
            return empty_value
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
        tIds = defaultdict(list)
        vIds = defaultdict(list)

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
