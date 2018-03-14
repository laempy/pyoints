import numpy as np

# TODO module description

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
        Record array similar to A, but with additional field B.    
    """
    assert isinstance(A, np.ndarray)
    if not isinstance(B, np.ndarray):
        B = np.array(B)
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
    A: `numpy.recarray`
        Numpy recarray to fuse.
    B: `numpy.recarray`
        Numpy recarray to fuse.
        
    Returns
    -------
    recarray: `numpy.recarray`
        Record array with same fields as A and B.    
    """
    assert isinstance(A, np.recarray)
    assert isinstance(B, np.recarray)
    assert A.shape == B.shape
    
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
        List of numpy.ndarrays to merge.
    
    Returns
    -------
    merged: `numpy.ndarray`
        Merged numpy record array.
    """
    assert hasattr(arrays,'__getitem__')
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
    """
    assert hasattr(func,'__call__')
    assert hasattr(ndarray,np.ndarray)
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
    
    assert hasattr(gen,'__iter__')
    assert hasattr(func,'__call__')
    
    values = [func(item) for item in gen]
    return np.array(values, dtype=dtype).view(np.recarray)


def recarray(dataDict, dtype=[]):
    """Converts a dictionary of array like objects to a numpy record array. 
    
    Parameters
    ----------
    dataDict: `numpy.recarray`
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
    ...    'numeric':  [1,3,1,2],
    ...    'missingvalues':  [None,None,'str',None],
    ... })
    >>> rec.dtype
    dtype((numpy.record, [('text', 'O'), ('coords', '<i8', (2,)), ('numeric', '<i8'), ('missingvalues', 'O')]))
    >>> print rec.coords
    [[3 4]
     [3 2]
     [0 2]
     [5 2]]
    >>> print rec[0]
    ('text1', [3, 4], 1, None)
    """

    assert hasattr(dataDict,'__getitem__') and hasattr(dataDict,'keys')
    
    # check data types
    dtype = np.dtype(dtype)
    for colName in dtype.names:
        assert colName in dataDict.keys(), 'column "%s" not found!' % colName

    # get datatypes
    outDtypes = []
    for colName in dataDict.keys():
        
        if colName not in dtype.names:
            dt = np.dtype(object)
            outDtype = (colName, dt) # default data type
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
    #print recarray
    assert isinstance(rec,(np.recarray,np.ndarray))
    
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
    assert hasattr(data,'__len__')
    strings = np.array(data,dtype=str)

    ismissing = np.equal(data, None)
    ismissing[strings == 'nan'] = True

    return ismissing


def colzip(arr):
    """ Splits a numpy array by each collumn.
    
    Parameters
    ----------
    arr : (n,k), np.ndarray
        Numpy array with `n` rows and `k` columns.
        
    Returns
    -------
    columns : list of np.ndarray
        List of k np.ndarrays 
    """
    
    assert isinstance(arr,np.ndarray)
    assert len(arr.shape) == 2
    
    cols = []
    for col in range(arr.shape[1]):
        cols.append(arr[:,col])
    return cols


def fields_view(arr,fields,dtype=None):
    if dtype is None:
        dtype = np.dtype({name:arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype, arr, 0, arr.strides)

