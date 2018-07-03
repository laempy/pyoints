# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:25:13 2018

@author: sebastian
"""

import plyfile
import numpy as np

from .. georecords import GeoRecords
from .. import nptools


def loadPly(infile, proj):
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

    return GeoRecords(proj, records)


def writePly(records, outfile):

    # create view
    dtypes = []
    for i, name in enumerate(records.dtype.names):
        if name == 'coords':
            dtypes.extend([('x', float), ('y', float), ('z', float)])
        else:
            dtypes.append(records.dtype.descr[i])
    records = records.view(dtypes)

    # change datatype if required (bug in plyfile)
    dtypes = []
    for i, name in enumerate(records.dtype.names):
        desc = list(records.dtype.descr[i])
        if desc[1] == '<i8':
            print('set')
            desc[1] = '<i4'
        dtypes.append(tuple(desc))
    records = records.astype(dtypes)

    # save data
    el = plyfile.PlyElement.describe(records.view(dtypes), 'vertex')
    ply = plyfile.PlyData([el], comments=['header comment'])
    ply.write(outfile)
