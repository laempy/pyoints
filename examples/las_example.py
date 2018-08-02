# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:51:38 2018

@author: sebastian
"""

import os
import numpy as np

from pointspy import (
    storage,
    projection,
    Extent,
    nptools,
    GeoRecords
)

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Create GeoRecords from scratch
proj = projection.Proj.from_epsg(25832)
center = np.array([332592.88, 5513244.80, 120])
data = nptools.recarray({
    'coords':  Extent([0, 0, 0, 1, 2, 3]).corners + center,
    'intensity': [3, 2, 4, 1, 5, 2, 5, 2],
})
geoRecords = GeoRecords(proj, data)
print(geoRecords.dtype)

# save LAS file
outfile = os.path.join(outpath, 'test.las')
print("save %s" % outfile)
storage.writeLas(geoRecords, outfile)


# load LAS file
lasReader = storage.LasReader(outfile)
print(lasReader.t.origin)
las = lasReader.load()
print(las.dtype)