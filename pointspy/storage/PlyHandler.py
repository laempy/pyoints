"""Hanlding of .ply-files.
"""

import plyfile
import numpy as np


def loadPly(infile):
    """Loads a .ply file.

    Parameters
    ----------
    infile : String
        PLY-file to be read.

    Returns
    -------
    np.recarray
        Loaded data.

    See Also
    --------
    writePly

    """
    plydata = plyfile.PlyData.read(infile)
    records = plydata['vertex'].data.view(np.recarray)

    # rename fields
    dtypes = [('coords', float, 3)]
    fields = [records.dtype.descr[i] for i in range(3, len(records.dtype))]
    dtypes.extend(fields)
    dtypes = np.dtype(dtypes)

    # change to propper names
    names = []
    for name in dtypes.names:
        names.append(name.replace('scalar_', ''))
    dtypes.names = names

    records = records.view(dtypes)

    return records


def writePly(records, outfile):
    """Saves data to a .ply file.

    Parameters
    ----------
    records : np.recarray
        Numpy record array to save.
    outfile : String
        Desired output .ply file .

    See Also
    --------
    loadPly


    """
    if not isinstance(records, np.recarray):
        raise TypeError("'records' needs to be a numpy record array")

    # create view
    dtypes = []
    for i, name in enumerate(records.dtype.names):
        if name == 'coords':
            dtypes.extend([('x', float), ('y', float), ('z', float)])
        else:
            dtypes.append(records.dtype.descr[i])
    records = records.view(dtypes)

    dtypes = []
    for i, name in enumerate(records.dtype.names):
        desc = list(records.dtype.descr[i])

        # change datatype if required (bug in plyfile?)
        if desc[1] == '<i8':
            desc[1] = '<i4'
        dtypes.append(tuple(desc))
    records = records.astype(dtypes)

    # save data
    el = plyfile.PlyElement.describe(records.view(dtypes), 'vertex')
    ply = plyfile.PlyData([el], comments=['header comment'])
    ply.write(outfile)
