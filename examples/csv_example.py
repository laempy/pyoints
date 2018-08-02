# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:03:44 2018

@author: sebastian
"""
import os
from pointspy import (
    storage,
)

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# Create GeoRecords from scratch
center = [332592.88, 5513244.80, 120]
geoRecords = storage.misc.create_random_GeoRecords(center=center, epsg=25832)

outfile = os.path.join(outpath, 'test.csv')
print('Save %s' % outfile)
storage.writeCsv(geoRecords, outfile)


# load file
print('load %s' % outfile)
data = storage.loadCsv(outfile, header=True)
print(data)
print(data.dtype)
