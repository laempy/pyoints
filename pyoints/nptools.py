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
"""Functions for convienient handling of numpy arrays.
"""

import numpy as np
from numbers import Number


def isarray(o):
    """Checks whether or nor an object is an array.

    Parameters
    ----------
    o : object
        Some object.

    Returns
    -------
    bool
        Indicates whether or not the object is an array.

    Examples
    --------

    >>> isarray([1, 2, 3])
    True
    >>> isarray('text')
    False

    """
    return (not isinstance(o, str) and
            hasattr(o, '__getitem__') and
            hasattr(o, '__iter__'))


def isnumeric(arr, dtypes=[np.uint8, np.uint16, np.int32, np.int64, np.float32, np.float64]):
    """Checks if the data type of an array is numeric.

    Parameters
    ----------
    arr : array_like
        Numpy array to check.
    dtypes : optional, tuple
        Tuple of allowed numeric data types.

    Returns
    -------
    bool
        Indicates weather or not the array is numeric.

    Raises
    ------
    TypeError

    Examples
    --------

    >>> isnumeric([1, 2, 3])
    True
    >>> isnumeric(['1', '2', '3'])
    False
    >>> isnumeric([1, 2, None])
    False

    """
    if not isarray(arr):
        raise TypeError("'arr' needs to an array like object")
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if not isinstance(dtypes, list):
        raise TypeError("'dtypes' needs to be a list. got %s" % (str(dtypes)))
    for dtype in dtypes:
        if np.issubdtype(arr.dtype.type, np.dtype(dtype).type):
            return True
    return False


def haskeys(d):
    """Checks if an object has keys and can be treated like a dictionary.

    Parameters
    ----------
    d : object
        Object to be checked.

    Returns
    -------
    bool
        Indicates weather or not the object has accessable keys.

    Examples
    --------

    >>> haskeys({'a': 5, 'b': 3})
    True
    >>> haskeys([5, 6])
    False
    >>> haskeys(np.recarray(3, dtype=[('a', int)]))
    False

    """
    return hasattr(d, '__getitem__') and hasattr(d, 'keys')


def missing(data):
    """Find missing values in an array.

    Parameters
    ----------
    data : array_like
        A array like object which might contain missing values. Missing values
        are assumed to be either None or NaN.

    Returns
    -------
    array_like(bool, shape=data.shape)
        Boolean values indicate missing values.

    Raises
    ------
    ValueError

    Examples
    --------

    Finding missing values in a list.

    >>> arr = ['str', 1, None, np.nan, np.NaN]
    >>> print(missing(arr))
    [False False  True  True  True]

    Finding missing values in a multi-dimensional array.

    >>> arr = np.array([(0, np.nan), (None, 1), (2, 3)], dtype=float)
    >>> print(missing(arr))
    [[False  True]
     [ True False]
     [False False]]

    """
    if not hasattr(data, '__len__'):
        raise ValueError("'data' has be a array like object")
    strings = np.array(data, dtype=str)

    ismissing = np.equal(data, None)
    ismissing[strings == 'nan'] = True

    return ismissing


def dtype_subset(dtype, names):
    """Creates a subset of a numpy type object.

    Parameters
    ----------
    dtype : list or np.dtype
        Numpy data type.
    names : list of str
        Fields to select.

    Raises
    ------
    TypeError

    Returns
    -------
    list
        Desired subest of numpy data type descriptions.

    Examples
    --------

    >>> dtypes = [('coords', float, 3), ('values', int), ('text', '<U0')]
    >>> print(dtype_subset(dtypes, ['text', 'coords']))
    [('text', '<U0'), ('coords', '<f8', (3,))]

    """
    if not hasattr(names, '__iter__'):
        raise TypeError("'names' needs to be iterable")
    descr = np.dtype(dtype).descr
    out_dtype = []
    for name in names:
        for dt in descr:
            if dt[0] == name:
                out_dtype.append(dt)
    return out_dtype


def recarray(data_dict, dtype=[], dim=1):
    """Converts a dictionary of array like objects to a numpy record array.
    This function is mostly used for convienience.

    Parameters
    ----------
    data_dict : dict
        Dictionary of array like objects to convert to a numpy record array.
        Each key in `data_dict` represents a field name of the output record
        array. Each item in `data_dict` represents the corresponding values.
        Thus at least `shape[0:dim]` of all input arrays in `data_dict` have
        to match.
    dtype : optional, numpy.dtype
        Describes the desired data types of specific fields. If the data dype
        of a field is not given, the data type is estimated from by numpy.
    dim : positive int
        Desired dimension of the resulting numpy record array.

    Returns
    -------
    np.recarray
        Numpy record array build from input dictionary.

    Raises
    ------
    TypeError, ValueError

    Examples
    --------

    Creation of a one dimensional numpy record array using a dictionary.

    >>> rec = recarray({
    ...    'coords': [ (3, 4), (3, 2), (0, 2), (5, 2)],
    ...    'text': ['text1', 'text2', 'text3', 'text4'],
    ...    'n':  [1, 3, 1, 2],
    ...    'missing':  [None, None, 'str', None],
    ... })
    >>> print(sorted(rec.dtype.names))
    ['coords', 'missing', 'n', 'text']
    >>> print(rec.coords)
    [[3 4]
     [3 2]
     [0 2]
     [5 2]]

    Create a two dimensional record array.

    >>> data = {
    ...    'coords': [
    ...                 [(2, 3.2, 1), (-3, 2.2, 4)],
    ...                 [(0, 1.1, 2), (-1, 2.2, 5)],
    ...                 [(-7, -1, 3), (9.2, -5, 6)]
    ...             ],
    ...    'values': [[1, 3], [4, 0], [-4, 2]]
    ... }
    >>> rec = recarray(data, dim=2)
    >>> print(rec.shape)
    (3, 2)
    >>> print(rec.coords)
    [[[ 2.   3.2  1. ]
      [-3.   2.2  4. ]]
    <BLANKLINE>
     [[ 0.   1.1  2. ]
      [-1.   2.2  5. ]]
    <BLANKLINE>
     [[-7.  -1.   3. ]
      [ 9.2 -5.   6. ]]]
    >>> print(rec.values)
    [[ 1  3]
     [ 4  0]
     [-4  2]]
    >>> print(sorted(rec.dtype.names))
    ['coords', 'values']

    """
    if not haskeys(data_dict):
        raise TypeError("'dataDict' has to be a dictionary like object")

    # check data types
    dtype = np.dtype(dtype)
    for key in dtype.names:
        if key not in data_dict.keys():
            raise ValueError('column "%s" not found!' % key)

    if not isinstance(dim, int) and dim > 0:
        raise ValueError("'dim' has to be an integer greater zero")

    # convert to numpy arrays if neccessary
    for key in data_dict.keys():
        if not isinstance(data_dict[key], (np.ndarray, np.recarray)):
            if key in dtype.names:
                dt = dtype_subset(dtype, [key])
                data_dict[key] = np.array(data_dict[key], dtype=dt, copy=False)
            else:
                data_dict[key] = np.array(data_dict[key], copy=False)

    # get data types
    out_dtypes = []
    for key in data_dict.keys():
        if key in dtype.names:
            dt = dtype_subset(dtype, [key])[0]
        else:
            arr = data_dict[key]
            dt = (key, arr.dtype.descr[0][1], arr.shape[dim:])
        out_dtypes.append(dt)

    # define array
    shape = next(iter(data_dict.values())).shape
    rec = np.recarray(shape[:dim], dtype=out_dtypes)
    if len(rec) > 0:
        for key in data_dict.keys():
            rec[key][:] = data_dict[key]

    return rec


def add_fields(arr, dtypes, data=None):
    """Adds additional fields to a numpy record array.

    Parameters
    ----------
    arr : np.recarray
        Numpy record array to add fields to.
    dtypes : np.dtype
        Data types of the new fields.
    data : optional, list of array_like
        Data values of the new fields. The shape of each array has to be
        compatible to arr.

    Returns
    -------
    np.recarray
        Record array similar to `A`, but with additional fields of type
        `dtypes` and values of `data`.

    Examples
    --------

    >>> A = recarray({'a': [0, 1, 2, 3]})
    >>> C = add_fields(A, [('b', float, 2), ('c', int)])
    >>> print(sorted(C.dtype.descr))
    [('a', '<i8'), ('b', '<f8', (2,)), ('c', '<i8')]

    >>> D = add_fields(A, [('d', int), ('e', str)], data=[[1, 2, 3, 4], None])
    >>> print(D)
    [(0, 1, '') (1, 2, '') (2, 3, '') (3, 4, '')]

    """
    if not isinstance(arr, np.recarray):
        raise TypeError("'arr' has to be an numpy record array")
    if data is not None and not hasattr(data, '__iter__'):
        raise ValueError("'data' has to be iterable")

    dtypes = np.dtype(dtypes)

    # check for duplicate fields
    for name in dtypes.names:
        if hasattr(arr, name):
            raise ValueError("can not overwrite attribute '%s'" % name)
        if name in arr.dtype.names:
            raise ValueError("field '%s' already exists" % name)

    newDtypes = arr.dtype.descr + dtypes.descr

    # set values
    rec = np.recarray(arr.shape, dtype=newDtypes)
    for name in arr.dtype.names:
        rec[name] = arr[name]

    # set new values
    if data is not None:
        for name, column in zip(dtypes.names, data):
            if column is not None:
                rec[name] = column

    return rec


def fuse(*recarrays):
    """Fuses multiple numpy record arrays of identical shape to one array.

    Parameters
    ----------
    \*recarrays : np.recarray
        Numpy record arrays to fuse.

    Returns
    -------
    np.recarray
        Record array with all fields of `recarrays`.

    Examples
    --------

    Fuse one dimensional arrays.

    >>> A = recarray({'a': [0, 1, 2, 3]})
    >>> B = recarray({'b': [4, 5, 6, 7]})
    >>> C = fuse(A, B)
    >>> print(C.shape)
    (4,)
    >>> print(C.dtype.names)
    ('a', 'b')

    Fuse multiple two dimensional arrays.

    >>> A = recarray({'a': [[0, 1], [2, 3]]}, dim = 2)
    >>> B = recarray({'b': [[4, 5], [6, 7]]}, dim = 2)
    >>> C = recarray({
    ...         'c1': [['c1', 'c2'], ['c3', 'c4']],
    ...         'c2': [[0.1, 0.2], [0.3, 0.3]],
    ...     }, dim = 2)

    >>> D = fuse(A, B, C)

    >>> print(sorted(D.dtype.names))
    ['a', 'b', 'c1', 'c2']
    >>> print(D.shape)
    (2, 2)
    >>> print(D.a)
    [[0 1]
     [2 3]]
    >>> print(D.c1)
    [['c1' 'c2']
     ['c3' 'c4']]
    >>> print(D.c2)
    [[0.1 0.2]
     [0.3 0.3]]

    """
    shape = None
    dtype = []
    for arr in recarrays:
        if not isinstance(arr, np.recarray):
            raise TypeError("all arrays have to be of type 'np.recarray'")
        dtype.extend(arr.dtype.descr)

        # check shape
        if shape is None:
            shape = arr.shape
        elif not arr.shape == shape:
            raise ValueError("all arrays have to have the same shape")

    # define array
    fused = np.recarray(shape, dtype=dtype)
    for arr in recarrays:
        for name in arr.dtype.names:
            fused[name] = arr[name]

    return fused


def merge(arrays, strategy=np.concatenate):
    """Merges multiple arrays with similar fields.

    Parameters
    ----------
    arrays : list of np.recarray
        Numpy arrays to merge.
    strategy : optional, function
        Aggregate function to apply during merging. Suggested values:
        np.conatenate, np.hstack, np.vstack, np.dstack.

    Returns
    -------
    np.recarray
        Merged numpy record array of same data type as the first input array.

    Raises
    ------
    TypeError

    Examples
    --------

    One dimensional arrays.

    >>> A = recarray({'a': [(0, 1), (2, 3), (4, 5)], 'b': ['e', 'f', 'g']})
    >>> B = recarray({'a': [(6, 7), (8, 9), (0, 1)], 'b': ['h', 'i', 'j']})
    >>> C = recarray({'a': [(2, 3), (4, 5), (6, 7)], 'b': ['k', 'l', 'm']})

    >>> D = merge((A, B, C))
    >>> print(sorted(D.dtype.names))
    ['a', 'b']
    >>> print(D.b)
    ['e' 'f' 'g' 'h' 'i' 'j' 'k' 'l' 'm']
    >>> print(D.shape)
    (9,)

    >>> D = merge((A, B, C), strategy=np.hstack)
    >>> print(D.shape)
    (9,)

    >>> D = merge((A, B, C), strategy=np.vstack)
    >>> print(D.shape)
    (3, 3)

    >>> D = merge((A, B, C), strategy=np.dstack)
    >>> print(D.shape)
    (1, 3, 3)

    Merge two dimensional arrays.

    >>> A = recarray({
    ...     'a': [(0, 1), (2, 3)], 'b': [('e', 'f'), ('g', 'h')]
    ... }, dim=2)
    >>> B = recarray({
    ...     'a': [(4, 5), (6, 7)], 'b': [('i', 'j'), ('k', 'l')]
    ... }, dim=2)
    >>> C = recarray({
    ...     'a': [(1, 3), (7, 2)], 'b': [('m', 'n'), ('o', 'p')]
    ... }, dim=2)
    >>> D = merge((A, B, C))

    >>> print(sorted(D.dtype.names))
    ['a', 'b']
    >>> print(D.b)
    [['e' 'f']
     ['g' 'h']
     ['i' 'j']
     ['k' 'l']
     ['m' 'n']
     ['o' 'p']]
    >>> print(D.shape)
    (6, 2)

    >>> D = merge((A, B, C), strategy=np.hstack)
    >>> print(D.shape)
    (2, 6)
    >>> D = merge((A, B, C), strategy=np.vstack)
    >>> print(D.shape)
    (6, 2)
    >>> D = merge((A, B, C), strategy=np.dstack)
    >>> print(D.shape)
    (2, 2, 3)
    >>> D = merge((A, B, C), strategy=np.concatenate)
    >>> print(D.shape)
    (6, 2)

    >>> A = np.recarray(1, dtype=[('a', object, 2), ('b', str)])
    >>> B = np.recarray(2, dtype=[('a', object, 2), ('b', str)])
    >>> D = merge((A, B), strategy=np.concatenate)
    >>> print(D)
    [([None, None], '') ([None, None], '') ([None, None], '')]

    """
    if not hasattr(arrays, '__iter__'):
        raise TypeError("'arrays' needs to be iterable")
    dtype = None
    for arr in arrays:
        if not isinstance(arr, (np.recarray, np.ndarray)):
            raise TypeError("'array' needs to be an iterable of 'np.recarray'")
        if dtype is None:
            dtype = arrays[0].dtype.descr
        elif not arr.dtype.descr == dtype:
            raise TypeError("all data types need to match")
    return arrays[0].__array_wrap__(strategy(arrays))


def flatten_dtypes(np_dtypes):
    """Exract name, datatype and shape information from a numpy data type.

    Parameters
    ----------
    np_dtypes : np.dtype
        Numpy data types to flatten.

    Returns
    -------
    names : list of str
        Names of fields.
    dtypes : list of dtypes
        Data types of fields.
    shapes : list of tuples
        Shapes of fields.

    Examples
    --------

    >>> dtype = np.dtype([
    ...     ('simple', int),
    ...     ('multidimensional', float, 3),
    ... ])
    >>> names, dtypes, shapes = flatten_dtypes(dtype)
    >>> names
    ['simple', 'multidimensional']
    >>> dtypes
    [dtype('int64'), dtype('float64')]
    >>> shapes
    [0, 3]

    """
    np_dtypes = np.dtype(np_dtypes)

    dtypes = []
    shapes = []
    names = list(np_dtypes.names)

    for name in names:

        dtype = np_dtypes[name]
        shape = 0

        subDtype = dtype.subdtype
        if subDtype is not None:
            shape = dtype.shape[0]
            dtype = dtype.subdtype[0]

        dtypes.append(dtype)
        shapes.append(shape)

    return names, dtypes, shapes


def unnest(arr, deep=False):
    """Unnest a numpy record array. This function recursively splits a nested
    numpy array to a list of arrays.

    Parameters
    ----------
    rec: np.recarray or np.ndarray
        Numpy array to unnest.
    deep : bool
        Indicates whether or not numpy ndarrays shall be splitted into
        individual colums or not.

    Raises
    ------
    TypeError

    Returns
    -------
    list
        List of unnested fields.

    Examples
    --------

    >>> dtype = [
    ...    ('regular', np.int, 1),
    ...    ('nested', [
    ...         ('child1', np.str),
    ...         ('child2', np.float, 2)
    ...    ])
    ... ]
    >>> rec = np.ones(2, dtype=dtype).view(np.recarray)
    >>> print(rec.nested.child2)
    [[1. 1.]
     [1. 1.]]

    >>> unnested = unnest(rec)
    >>> print(unnested[0])
    [1 1]
    >>> print(unnested[1])
    ['' '']
    >>> print(unnested[2])
    [[1. 1.]
     [1. 1.]]

    """
    if not isinstance(arr, (np.recarray, np.ndarray)):
        m = "'rec' has to be an instance of 'np.recarray' or 'np.ndarray'"
        raise TypeError(m)
    if not isinstance(arr, np.recarray):
        if deep and len(arr.shape) > 1:
            ret = []
            for col in colzip(arr):
                ret.extend(unnest(col, deep=deep))
        else:
            ret = [arr]
    else:
        ret = []
        for name in arr.dtype.names:
            ret.extend(unnest(arr[name], deep=deep))
    return ret


def colzip(arr):
    """Splits a two dimensional numpy array into a list of columns.

    Parameters
    ----------
    arr : np.ndarray(shape=(n, k)) or np.recarray(shape=(n, ))
        Numpy array with `n` rows and `k` columns.

    Returns
    -------
    columns : list of np.ndarray
        List of `k` numpy arrays.

    Raises
    ------
    TypeError, ValueError

    Examples
    --------

    >>> arr = np.eye(3, dtype=int)
    >>> cols = colzip(arr)
    >>> len(cols)
    3
    >>> print(cols[0])
    [1 0 0]

    """
    if isinstance(arr, np.recarray):
        return [arr[name] for name in arr.dtype.names]
    elif isinstance(arr, np.ndarray):
        if not len(arr.shape) == 2:
            raise ValueError("'arr' has be two dimensional")
        return [arr[:, col] for col in range(arr.shape[1])]
    else:
        raise TypeError("unexpected type of 'arr'")


def apply_function(arr, func, dtype=None):
    """Applies a function to each record of a numpy array.

    Parameters
    ----------
    arr : np.ndarray or np.recarray
        Numpy array to apply function to.
    func : function
        Function to apply to each record.
    dtypes : optional, np.dtype
        Desired data type of the output array.

    Returns
    -------
    np.recarray
        Record array similar to input array, but with function applied to.

    Examples
    --------

    Apply a function to a numpy ndarray.

    >>> arr = np.ones((2, 3), dtype=[('a', int), ('b', int)])
    >>> func = lambda item: item[0] + item[1]
    >>> print(apply_function(arr, func))
    [[2 2 2]
     [2 2 2]]

    Aggregate a one dimensional numpy reccord array.

    >>> data = { 'a': [0, 1, 2, 3], 'b': [1, 2, 3, 4] }
    >>> arr = recarray(data)
    >>> func = lambda record: record.a + record.b
    >>> print(apply_function(arr, func))
    [1 3 5 7]

    Two dimensional case.

    >>> data = { 'a': [[0, 1], [2, 3]], 'b': [[1, 2], [3, 4]] }
    >>> arr = recarray(data, dim=2)
    >>> func = lambda record: record.a ** record.b
    >>> print(apply_function(arr, func))
    [[ 0  1]
     [ 8 81]]

    Specify the output data type.

    >>> func = lambda record: (record.a + record.b, record.a ** record.b)
    >>> print(apply_function(arr, func, dtype=[('c', float), ('d', int)]))
    [[(1.,  0) (3.,  1)]
     [(5.,  8) (7., 81)]]

    Specify a multi-dimensional output data type.

    >>> func = lambda record: (record.a + 2, [record.a ** 2, record.b * 3])
    >>> print(apply_function(arr, func, dtype=[('c', float), ('d', int, 2)]))
    [[(2., [ 0,  3]) (3., [ 1,  6])]
     [(4., [ 4,  9]) (5., [ 9, 12])]]

    >>> func = lambda record: ([record.a ** 2, record.b * 3],)
    >>> print(apply_function(arr, func, dtype=[('d', int, 2)]))
    [[([ 0,  3],) ([ 1,  6],)]
     [([ 4,  9],) ([ 9, 12],)]]

    """
    if not callable(func):
        raise ValueError("'func' needs to be callable")
    if not isinstance(arr, (np.ndarray, np.recarray)):
        m = "'ndarray' needs to an instance of 'np.ndarray' or 'np.recarray'"
        raise TypeError(m)
    if dtype is not None:
        dtype = np.dtype(dtype)

    args = np.broadcast(None, arr)
    values = [func(*arg[1:]) for arg in args]
    if dtype is None or dtype.names is None:
        res = np.array(values, dtype=dtype).reshape(arr.shape)
    else:
        res = np.array(
            values, dtype=dtype).reshape(arr.shape).view(np.recarray)
    return res


def indices(shape, flatten=False):
    """Create keys or indices of a numpy ndarray.

    Parameters
    ----------
    shape : array_like(int)
        Shape of desired output array.

    Returns
    -------
    np.ndarray(int, shape=(\*shape, len(shape)))
        Array of indices with desired `shape`. Each entry provides a index
        tuple to access the array entries.

    Examples
    --------

    One dimensional case.

    >>> keys = indices(9)
    >>> print(keys.shape)
    (9,)
    >>> print(keys)
    [0 1 2 3 4 5 6 7 8]

    Two dimensional case.

    >>> keys = indices((3, 4))
    >>> keys.shape
    (3, 4, 2)
    >>> print(keys)
    [[[0 0]
      [0 1]
      [0 2]
      [0 3]]
    <BLANKLINE>
     [[1 0]
      [1 1]
      [1 2]
      [1 3]]
    <BLANKLINE>
     [[2 0]
      [2 1]
      [2 2]
      [2 3]]]

    Get iterable of indices.

    >>> keys = indices((3, 4), flatten=True)
    >>> print(keys)
    [[0 0]
     [0 1]
     [0 2]
     [0 3]
     [1 0]
     [1 1]
     [1 2]
     [1 3]
     [2 0]
     [2 1]
     [2 2]
     [2 3]]

    """
    if isinstance(shape, int):
        keys = np.arange(shape)
    else:
        shape = np.array(shape, dtype=int)
        keys = np.indices(shape)
        if flatten:
            keys = keys.reshape(-1, np.product(shape)).T
        else:
            keys = np.moveaxis(keys, 0, -1)
    return keys


def range_filter(arr, min_value=-np.inf, max_value=np.inf):
    """Filter values by range.

    Parameters
    ----------

    arr : array_like(Number)
        Numeric array to filter.
    min_value,max_value : Number
        Minimum and maximum values to define the desired value range
        `[min_value, max_value]` of `arr`.

    Returns
    -------
    np.ndarray(int)
        Indices of all values of `arr` in the desired range.

    Examples
    --------

    Filter a one dimensional array.

    >>> a = [0, 2, 1, -1, 5, 7, 9, 4, 3, 2, -2, -11]

    >>> indices = range_filter(a, min_value=0)
    >>> print(indices)
    [0 1 2 4 5 6 7 8 9]

    >>> indices = range_filter(a, max_value=5)
    >>> print(indices)
    [ 0  1  2  3  4  7  8  9 10 11]

    >>> indices = range_filter(a, min_value=0, max_value=5)
    >>> print(indices)
    [0 1 2 4 7 8 9]
    >>> print(np.array(a)[indices])
    [0 2 1 5 4 3 2]

    Filter a multi-dimensional array.

    >>> a = [(1, 0), (-2, -1), (3, -5), (4, 2), (-7, 9), (0.5, 2)]

    >>> indices = range_filter(a, min_value=2)
    >>> print(indices)
    ((2, 3, 3, 4, 5), (0, 0, 1, 1, 1))
    >>> print(np.array(a)[indices])
    [3. 4. 2. 9. 2.]

    >>> indices = range_filter(a, min_value=2, max_value=5)
    >>> print(indices)
    ((2, 3, 3, 5), (0, 0, 1, 1))
    >>> print(np.array(a)[indices])
    [3. 4. 2. 2.]

    """
    if not isnumeric(arr):
        raise TypeError("'arr' needs to be an numeric array")
    if not isinstance(min_value, Number):
        raise TypeError("'min_value' needs to a number")
    if not isinstance(max_value, Number):
        raise TypeError("'max_value' needs to a number")
    if not max_value >= min_value:
        m = "'max_value' needs to be greater or equal 'min_value'"
        raise ValueError(m)

    arr = np.array(arr)
    mask = np.all((arr >= min_value, arr <= max_value), axis=0)
    if len(arr.shape) == 1:
        ids = np.where(mask)[0]
    else:
        ids = tuple(map(tuple, np.array(np.where(mask))))

    return ids



def max_value_range(arr):
    """Returns the maximum value range of a numeric numpy array.
    
    Parameters
    ----------
    arr : np.ndarray(Number)
        Array to derive allowed value range for.
        
    Returns
    -------
    min_value,max_value : int
        Minimum and maximum value
        
    Examples
    --------
    
    >>> arr = np.array([3, 4, 2, 1])
    
    >>> value_range = max_value_range(arr.astype(np.uint8))
    >>> print(value_range)
    (0, 255)
    
    >>> value_range = max_value_range(arr.astype(np.uint16))
    >>> print(value_range)
    (0, 65535)
    
    >>> value_range = max_value_range(arr.astype(np.int8))
    >>> print(value_range)
    (-128, 127)
    
    >>> value_range = max_value_range(arr.astype(np.int16))
    >>> print(value_range)
    (-32768, 32767)
            
    >>> value_range = max_value_range(arr.astype(np.float16))
    >>> print(value_range)
    (-65500.0, 65500.0)
    
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("'arr' needs to be a numpy array")
    dtype = arr.dtype
    if dtype.kind in ('i', 'u'):
        info = np.iinfo(dtype)
    elif dtype.kind in ('f'):
        info = np.finfo(dtype)
    else:
        raise ValueError("unknown data type '%s'" % dtype)
    
    return info.min, info.max
