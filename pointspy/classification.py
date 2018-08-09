# BEGIN OF LICENSE NOTE
# This file is part of Pointspy.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
#
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Collection of functions to classify or reclassify values or cluster values.
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
        Array of class indices.
    ids : optional, array_like(int, shape=(n))
        Indices to specify a subset of `classification`. If None, the indices
        are numbered consecutively.
    min_size,max_size : optional, positive int
        Minimum and maximum desired size of a class to be kept in the result.
    missing_value : optional, object
        Default value for unclassified values.

    Returns
    -------
    dict
        Dictionary of class indices. The dictionary keys represent the class
        ids, while the values represent the indices in the original array.

    See Also
    --------
    dict_to_classes
        Dictionary representation of `classification`.

    Examples
    --------

    >>> classes = ['cat', 'cat', 'dog', 'bird', 'dog', 'bird', 'cat', 'dog']
    >>> class_dict = classes_to_dict(classes)
    >>> print(sorted(class_dict))
    ['bird', 'cat', 'dog']
    >>> print(class_dict['cat'])
    [0, 1, 6]

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
    min_size,max_size : optional, positive int
        Minimum and maximum desired size of a class to be kept in the result.
    missing_value : optional, object
        Default value for unclassified values.

    Returns
    -------
    np.ndarray(int, shape=(n))
        Array representation of `classes_dict`.

    See Also
    --------
    classes_to_dict

    Notes
    -----
    Only a minimal input validation is provided.

    Examples
    --------

    Alphanumeric classes.

    >>> classes_dict = {'bird': [0, 1, 5, 4], 'dog': [3, 6, 8], 'cat': [7]}
    >>> print(dict_to_classes(classes_dict, 10, missing_value=''))
    ['bird' 'bird' '' 'dog' 'bird' 'bird' 'dog' 'cat' 'dog' '']

    Omit small classes.

    >>> print(dict_to_classes(classes_dict, 10, min_size=2))
    ['bird' 'bird' -1 'dog' 'bird' 'bird' 'dog' -1 'dog' -1]

    Numeric classes.

    >>> classes_dict = {0: [0, 1, 5], 1: [3, 6], 2: [7, 2]}
    >>> print(dict_to_classes(classes_dict, 9))
    [0 0 2 1 -1 0 1 2 -1]

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
    for cId, ids in classes_dict.items():
        if len(ids) >= min_size and len(ids) <= max_size:
            classification[ids] = cId

    return classification


def split_by_breaks(values, breaks):
    """Classifiy values by ranges.

    Parameters
    ----------
    values : array_like(Number, shape=(n))
        Values to classify.
    breaks : array_like(Number, shape=(m))
        Series of value ranges.

    Returns
    -------
    classification : np.ndarray(int, shape=(n))
        Desired class affiliation of `values`. A value of `classification[i]`
        equal to `k` means that 'values[i]' is in range
        `[breaks[k], breaks[k][`

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
        Dictionary to rename.
    ids : optional, array_like(shape=(len(d)))
        Desired key names. If None, the keys are numbered consecutively.

    Returns
    -------
    dict
        Dictionary with new names.

    Examples
    --------

    >>> d = {1: [0, 1], 2: None, 3: 'text'}
    >>> renamed_dict = rename_dict(d, ['first', 'second', 'last'])
    >>> print(sorted(renamed_dict))
    ['first', 'last', 'second']

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
    empty_value : optional, object
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
