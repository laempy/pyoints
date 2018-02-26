import numpy as np
import warnings
from .. import nptools


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
                data = nptools.merge([data, records])

            skip += bulk

    return data