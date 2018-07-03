import os
import numpy as np
import warnings

from .. import nptools


def loadCsv(fileName, sep=';', bulk=100000, dtype=None, header=True):

    skip = 1 if header else 0

    if dtype is None:
        with open(fileName) as f:
            header_names = f.readline().strip(os.linesep).split(sep)
            dtype = [(name, float) for name in header_names]

    records = np.empty(0, dtype=dtype).view(np.recarray)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        while True:
            try:
                data = np.genfromtxt(
                    fileName,
                    usecols=None,
                    delimiter=sep,
                    skip_header=skip,
                    max_rows=bulk,
                    dtype=dtype,
                    invalid_raise=True
                ).view(np.recarray)
            except (Warning, StopIteration):
                break
            if len(data.shape) == 0:
                data = np.array([data], dtype=dtype).view(np.recarray)

            records = nptools.merge([records, data])
            skip += bulk

    return records


def writeCsv(data, filename, sep=';', header=None):
    if header is None and hasattr(
            data, 'dtype') and data.dtype.names is not None:
        header = sep.join(data.dtype.names)
    if header is None:
        header = ''
    np.savetxt(
        filename,
        data,
        fmt='%s',
        delimiter=sep,
        header=header,
        comments='')
