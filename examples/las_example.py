# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:51:38 2018

@author: sebastian
"""

import os

from pointspy import storage


outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Create GeoRecords from scratch
center = [332592.88, 5513244.80, 120]
geoRecords = storage.misc.create_random_GeoRecords(center=center, epsg=25832)
print(geoRecords.dtype)

# save LAS file
outfile = os.path.join(outpath, 'test.las')
print("save %s" % outfile)
storage.writeLas(geoRecords, outfile)

# load LAS file
print("load %s" % outfile)
lasReader = storage.LasReader(outfile)
print(lasReader.t.origin)
las = lasReader.load()
print(las.dtype)
print(las[0])
