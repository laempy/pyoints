import numpy as np
from numpy.lib.recfunctions import merge_arrays
from numpy.lib.recfunctions import stack_arrays
from numpy.lib.recfunctions import append_fields
from numpy.lib import recfunctions
import pandas
import warnings


def addField(A, B, name):
    B = np.array(B)
    dtype = A.dtype.descr
    dtype.append((name, B.dtype))
    arr = np.recarray(A.shape, dtype=dtype)
    for colName in A.dtype.names:
        arr[colName] = A[colName]
    arr[name] = B
    return arr
    # shape=A.shape
    # return
    # recfunctions.rec_append_fields(A.flatten(),(name),B.flatten()).reshape(shape)


def fuse(A, B):
    dtype = A.dtype.descr
    dtype.extend(B.dtype.descr)

    retArray = np.recarray(A.shape, dtype=dtype)
    for name in A.dtype.names:
        retArray[name] = A[name]
    for name in B.dtype.names:
        retArray[name] = B[name]

    return retArray


def merge(arrays):
    return arrays[0].__array_wrap__(np.hstack(arrays))


def mergeCols(arrays, dtype=None):
    array = np.hstack(arrays)
    if dtype is not None:
        array = array.astype(dtype)
    return array.view(np.recarray)


def flattenDtypes(npDtypes):

    dtypes = []
    shapes = []
    names = npDtypes.names

    for name in names:

        dtype = npDtypes[name]
        shape = 0

        subDtype = dtype.subdtype
        if subDtype is not None:
            shape = dtype.shape[0]
            dtype = dtype.subdtype[0]

        dtypes.append(dtype)
        shapes.append(shape)

    return names, dtypes, shapes


def loadCsv(fileName, dtype, skip=0, sep=';', cols=None, bulk=500000):
    data = None
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        while True:
            try:
                records = np.genfromtxt(
                    fileName,
                    usecols=cols,
                    delimiter=sep,
                    skip_header=skip,
                    max_rows=bulk,
                    dtype=dtype,
                    invalid_raise=True).view(
                    np.recarray)
            except (Warning, StopIteration):
                break
            if len(records.shape) == 0:
                records = np.array([records], dtype=dtype).view(np.recarray)

            if data is None:
                data = records
            else:
                data = merge([data, records])

            skip += bulk

    return data


def map(func, npArray, dtypes=None):
    args = np.broadcast(None, npArray)
    values = [func(*arg[1:]) for arg in args]
    res = np.array(
        values,
        dtype=dtypes).reshape(
        npArray.shape).view(
            np.recarray)
    return res


def aggregate(gen, func, dtype=None):
    values = [func(item) for item in gen]
    return np.array(values, dtype=dtype).view(np.recarray)


def recarray(dataDict, dtype=[]):
    dtype = np.dtype(dtype)

    for colName in dtype.names:
        assert colName in dataDict.keys(), 'column "%s" of npDtypes not found!' % colName

    # get datatypes
    outDtypes = []
    for colName in dataDict.keys():
        if colName not in dtype.names:

            # Find not empty row
            for i in range(len(dataDict[colName])):
                row = dataDict[colName][i]
                if row is not None:
                    break
            row = np.array(row)
            s = row.shape
            s = s if len(s) > 0 else 1
            dt = row.dtype
            if np.issubdtype(dt, np.str):
                dt = np.object_
            outDtype = (colName, dt, s)
        else:
            dt = dtype[colName]
            outDtype = (colName, dt)
        outDtypes.append(outDtype)

    recarray = np.rec.array(zip(*dataDict.values()),
                            names=dataDict.keys(), dtype=outDtypes)
    return recarray


def unnest(recarray):
    if recarray.dtype.names is None:
        return [recarray]
    else:
        ret = []
        for name in recarray.dtype.names:
            ret.extend(unnest(recarray[name]))
        return ret


def mergeColumns(recarray, dtype=None):

    if recarray.dtype.names is None:
        ret = np.copy(recarray)
        if dtype is not None:
            ret = ret.astype(dtype)
        return ret

    # Shape
    s = 0
    for name in recarray.dtype.names:
        shape = recarray.dtype[name].shape
        s = s + 1 if len(shape) == 0 else s + shape[0]

    ret = np.empty((len(recarray), s), dtype=dtype)
    k = 0
    for name in recarray.dtype.names:
        shape = recarray.dtype[name].shape
        if len(shape) == 0:
            ret[:, k] = recarray[name]
            k += 1
        else:
            j = k + shape[0]
            ret[:, k:j] = recarray[name]
            k = j

    return ret


def isMissing(data):
    mask = np.equal(data, None)
    try:
        mask[np.isnan(data)] = True
    except TypeError:
        pass
    return mask
