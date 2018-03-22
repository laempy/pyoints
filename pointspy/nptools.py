"""Provides functions for convienient handling of numpy arrays.
"""

import numpy as np


def isarray(o):
    """Checks weather or nor an object is an array.

    Parameters
    ----------
    o: object
        Some object.

    Returns
    -------
    bool
        Indicates weather or not the object is an array.

    Example
    -------

    >>> isarray([1,2,3])
    True
    >>> isarray('str')
    False

    """
    return hasattr(o, '__getitem__') and hasattr(o, '__iter__')


def isnumeric(arr, dtypes=[np.int64, np.float64]):
    """Checks if the data type of an `numpy.ndarray` is numeric.

    Parameters
    ----------
    arr : np.ndarray
        Numpy array to add field to.
    dtypes : optional, tuple
        Tuple of data types.

    Returns
    -------
    bool
        Indicates weather or not the array is numeric.

    Example
    -------

    >>> isnumeric([1,2,3])
    True
    >>> isnumeric(['1','2','3'])
    False
    >>> isnumeric([1,2,None])
    False

    """
    if not isarray(arr):
        raise ValueError("'arr' needs to an array like object")
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if not isinstance(dtypes, list):
        raise ValueError("'dtypes' needs to be a list. got %s" % (str(dtypes)))
    for dtype in dtypes:
        if np.issubdtype(arr.dtype.type, dtype):
            return True
    return False


def add_field(A, B, name):
    """Adds field to array.

    Parameters
    ----------
    A: `numpy.ndarray`
        Numpy array to add field to.
    B: `array_like`
        Field attributes.
    name: `str`
        Name of new field.

    Returns
    -------
    recarray: `numpy.recarray`
        Record array similar to `A`, but with additional field `B`.

    Examples
    --------
    TODO

    """
    if not isinstance(A, np.ndarray):
        raise ValueError("'A' has to be of type 'numpy.ndarray'")
    if not hasattr(B, '__len__'):
        raise ValueError("'B' has to have a length")
    if not isinstance(B, np.ndarray):
        B = np.array(B)
    if not A.shape == B.shape:
        raise ValueError("'A' has to have the same shape as 'B'")

    dtype = A.dtype.descr
    dtype.append((name, B.dtype))
    rec = np.recarray(A.shape, dtype=dtype)
    for colName in A.dtype.names:
        rec[colName] = A[colName]
    rec[name] = B
    return rec


def fuse(A, B):
    """Fuses two numpy record arrays to one array.

    Parameters
    ----------
    A, B: `numpy.recarray`
        Numpy recarrays to fuse.

    Returns
    -------
    recarray: `numpy.recarray`
        Record array with same fields as `A` and `B`.

    Examples
    --------
    TODO

    """
    if not isinstance(A, np.recarray):
        raise ValueError("'A' has to be of type 'numpy.recarray'")
    if not isinstance(B, np.recarray):
        raise ValueError("'B' has to be of type 'numpy.recarray'")
    if not A.shape == B.shape:
        raise ValueError("'A' has to have the same shape as 'B'")

    dtype = A.dtype.descr
    dtype.extend(B.dtype.descr)

    fused = np.recarray(A.shape, dtype=dtype)
    for name in A.dtype.names:
        fused[name] = A[name]
    for name in B.dtype.names:
        fused[name] = B[name]

    return fused


def merge(arrays):
    """Merges multiple arrays.

    Parameters
    ----------
    arrays: `array_like`
        List of `numpy.ndarray`'s to merge.

    Returns
    -------
    merged: `numpy.ndarray`
        Merged numpy record array.

    Examples
    --------
    TODO

    """
    if not isinstance(arrays, (tuple, list)):
        raise ValueError("attribute 'arrays' has to be a list or tuple")
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            ValueError(
                "all elements of 'arrays' have to be of type 'numpy.recarray'")

    return arrays[0].__array_wrap__(np.hstack(arrays))


def flatten_dtypes(np_dtypes):
    """Exract name, datatype and shape information from numpy data type.

    Parameters
    ----------
    np_dtypes: `numpy.dtype`
        Numpy data types to flatten.

    Returns
    -------
    names: `list of str`
        Names of fields.
    dtypes: `list of dtypes`
        Data types of fields.
    shapes: `list of tuples`
        Shapes of fields.

    Examples
    --------

    >>> dtype = np.dtype([
    ...     ('simple',int),
    ...     ('multidimensional',float,3),
    ... ])
    >>> names, dtypes, shapes = flatten_dtypes(dtype)
    >>> names
    ['simple', 'multidimensional']
    >>> dtypes
    [dtype('int64'), dtype('float64')]
    >>> shapes
    [0, 3]

    """

    # ensure data type
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


def map_function(func, ndarray, dtypes=None):
    """Maps a function to each cell of a numpy array.

    Parameters
    ----------
    func: `function`
        Function to apply to each cell.
    ndarray: `numpy.ndarray`
        Numpy array to map function to.
    dtypes: optional, `numpy.dtype`
        Desired data type of return array.

    Returns
    -------
    recarray: `numpy.recarray`
        Record array similar to input array, but with function applied to.


    Examples
    --------
    TODO

    """
    assert hasattr(func, '__call__')
    assert hasattr(ndarray, np.ndarray)
    dtypes = np.dtype(dtypes)

    args = np.broadcast(None, ndarray)
    values = [func(*arg[1:]) for arg in args]
    res = np.array(
        values,
        dtype=dtypes)
    res = res.reshape(ndarray.shape)
    return res.view(np.recarray)


def aggregate(gen, func, dtype=None):
    """Aggregates TODO

    Parameters
    ----------
    gen: `iterable`
        Iterable object.
    func: `function`
        Function which aggregates the fields.
    dtype: optional, `numpy.dtype`
        Desired data type of return array.

    Returns
    -------
    recarray: `numpy.recarray`
        Record array similar to input array, but with function applied to.

    Examples
    --------
    TODO


    """

    assert hasattr(gen, '__iter__')
    assert hasattr(func, '__call__')

    values = [func(item) for item in gen]
    return np.array(values, dtype=dtype).view(np.recarray)


def recarray(dataDict, dtype=[]):
    """Converts a dictionary of array like objects to a numpy record array.

    Parameters
    ----------
    dataDict: `dict`
        Dictionary of array like objects to convert to a numpy record array.
    dtype: optional, `numpy.dtype`
        Describes the desired data type of specific fields.

    Returns
    -------
    recarray: `numpy.recarray`
        Numpy record array build from input dictionary.


    Examples
    --------

    Creation of an numpy record array using a dictionary.

    >>> rec = recarray({
    ...    'coords': [ (3,4), (3,2), (0,2), (5,2)],
    ...    'text': ['text1','text2','text3','text4'],
    ...    'n':  [1,3,1,2],
    ...    'missing':  [None,None,'str',None],
    ... })
    >>> rec.dtype.descr
    [('text', '|O'), ('missing', '|O'), ('coords', '<i8', (2,)), ('n', '<i8')]
    >>> print rec.coords
    [[3 4]
     [3 2]
     [0 2]
     [5 2]]
    >>> print rec[0]
    ('text1', None, [3, 4], 1)

    """
    if not (hasattr(dataDict, '__getitem__') and hasattr(dataDict, 'keys')):
        raise ValueError("'dataDict' has to be a dictionary like object")

    # check data types
    dtype = np.dtype(dtype)
    for colName in dtype.names:
        if colName not in dataDict.keys():
            raise ValueError('column "%s" not found!' % colName)

    # get datatypes
    outDtypes = []
    for colName in dataDict.keys():

        if colName not in dtype.names:
            dt = np.dtype(object)
            outDtype = (colName, dt)  # default data type
            # Find non empty row
            for row in dataDict[colName]:
                if row is not None:
                    row = np.array(row)
                    s = row.shape
                    if not np.dtype(str) == row.dtype.type:
                        dt = row.dtype
                    outDtype = (colName, dt, s)
                    break
        else:
            dt = dtype[colName]
            outDtype = (colName, dt)
        outDtypes.append(outDtype)
    rec = np.rec.array(zip(*dataDict.values()),
                       names=dataDict.keys(), dtype=outDtypes)
    return rec


def unnest(rec):
    """Unnest a numpy record array. Recursively adds each named field to a list.

    Parameters
    ----------
    rec: `numpy.recarray`
        Numpy record array to unnest.

    Returns
    -------
    unnested: `list`
        List of unnested fields.

    Examples
    --------

    >>> dtype = [
    ...    ('regular',np.int,1),
    ...    ('nested',[
    ...         ('child1','|S0'),
    ...         ('child2',np.float,2)
    ...    ])
    ... ]
    >>> rec = np.ones(2,dtype=dtype).view(np.recarray)
    >>> print rec.nested.child2
    [[1. 1.]
     [1. 1.]]
    >>> unnested = unnest(rec)
    >>> print unnested[0]
    [1 1]
    >>> print unnested[1]
    ['' '']
    >>> print unnested[2]
    [[1. 1.]
     [1. 1.]]

    """

    if not isinstance(rec, (np.recarray, np.ndarray)):
        raise ValueError("'rec' has to a 'np.recarray' or 'np.ndarray'")

    if rec.dtype.names is None:
        ret = [rec]
    else:
        ret = []
        for name in rec.dtype.names:
            ret.extend(unnest(rec[name]))
    return ret


def missing(data):
    """Find missing values.

    Parameters
    ----------
    data: `array_like`
        A array like object to search missing values for. Missing values are
        either None or NaN values.

    Returns
    -------
    missing: boolean numpy.ndarray
        Boolean values indicate missing values.

    Examples
    --------

    Finding missing values in a list.

    >>> arr = ['str',1,None,np.nan,np.NaN]
    >>> print missing(arr)
    [False False  True  True  True]

    Finding missing values in multidimensional arrays.

    >>> arr = np.array([(0,np.nan),(None,1),(2,3)],dtype=float)
    >>> print missing(arr)
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


def colzip(arr):
    """ Splits a two dimensional np.ndarray into a list of columns.

    Parameters
    ----------
    arr : (n,k), np.ndarray
        Numpy array with `n` rows and `k` columns.

    Returns
    -------
    columns : list of np.ndarray
        List of k np.ndarrays

    Examples
    --------
    TODO

    """
    if not (isinstance(arr, np.ndarray) and len(arr.shape) == 2):
        raise ValueError("'arr' has be a two dimensional 'np.ndarray'")

    cols = []
    for col in range(arr.shape[1]):
        cols.append(arr[:, col])
    return cols


def fields_view(arr, fields, dtype=None):
    """TODO

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    Examples
    --------
    TODO

    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("'arr' has be a 'np.ndarray'")

    if dtype is None:
        dtype = np.dtype({name: arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype, arr, 0, arr.strides)
