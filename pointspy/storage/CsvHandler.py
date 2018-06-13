import numpy as np

from .. import nptools



def loadCsv(fileName, sep=';', bulk=1000000, dtype=None, header=True):

    if header:
        skip = 1
    else:
        skip = 0

    if dtype is None:
        with open(fileName) as f:
            headerNames = f.readline().strip(os.linesep).split(sep)
        if dtype is None:
            dtype = [(name, float) for name in headerNames]
        #assert len(headerNames)==len(dtype)

    # records=np.loadtxt(fileName,delimiter=sep,skiprows=1,names=names).view(np.recarray)
    # records=np.genfromtxt(fileName,delimiter=sep,skip_header=1,names=names).view(np.recarray)
    # records=np.genfromtxt(fileName,delimiter=sep,names=True).view(np.recarray)
    records = nptools.loadCsv(fileName, dtype, skip=skip, bulk=bulk, sep=sep)
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