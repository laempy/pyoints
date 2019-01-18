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
"""Handling of .ply-files.
"""

import os
import plyfile
import numpy as np

from ..georecords import LasRecords
from ..projection import Proj

from ..misc import *
import warnings

def loadPly(infile, proj=Proj()):
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
    if not os.path.isfile(infile):
        raise IOError('file "%s" not found' % infile)

    with warnings.catch_warnings():
        # ignore UserWarning
        warnings.filterwarnings("ignore", category=UserWarning)
        plydata = plyfile.PlyData.read(infile)

    records = plydata['vertex'].data.view(np.recarray)

    # rename fields
    dtypes = [('coords', records.x.dtype, 3)]
    fields = [records.dtype.descr[i] for i in range(3, len(records.dtype))]
    dtypes.extend(fields)
    dtypes = np.dtype(dtypes)

    # change to propper names
    names = []
    for name in dtypes.names:
        names.append(name.replace('scalar_', ''))
    dtypes.names = names
    records = records.view(dtypes).copy()

    return LasRecords(proj, records)


def writePly(rec, outfile):
    """Saves data to a .ply file.

    Parameters
    ----------
    rec : np.recarray
        Numpy record array to save.
    outfile : String
        Desired output .ply file .

    See Also
    --------
    loadPly

    """
    if not isinstance(rec, np.recarray):
        raise TypeError("'records' needs to be a numpy record array")

    # create view
    dtypes = []
    for i, name in enumerate(rec.dtype.names):
        if name == 'coords':
            dtypes.extend([('x', float), ('y', float), ('z', float)])
        else:
            dtypes.append(rec.dtype.descr[i])
    rec = rec.view(dtypes)

    dtypes = []
    for i, name in enumerate(rec.dtype.names):
        desc = list(rec.dtype.descr[i])

        # change datatype if required (bug in plyfile?)
        if desc[1] == '<i8':
            desc[1] = '<i4'
        if desc[1] == '<u8':
            desc[1] = 'uint8'
        dtypes.append(tuple(desc))

    # save data
    el = plyfile.PlyElement.describe(rec.astype(dtypes), 'vertex')
    ply = plyfile.PlyData([el], comments=['created by "PoYnts"'])
    ply.write(outfile)
