# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:52:39 2018

@author: sebastian
"""

from pointspy import (
    storage,
    transformation,
)

infile = '/home/sebastian/Downloads/SP27GTIF.TIF'
infile = '/home/sebastian/Downloads/cea.tif'

outfile = '/home/sebastian/Schreibtisch/temp/test.tif'
rasterHandler = storage.RasterReader(infile)

grid = rasterHandler.load()

print grid.t
print rasterHandler.date

storage.writeRaster(grid, outfile)


grid.t = grid.t * transformation.s_matrix([2, 2])
print grid.t
storage.writeRaster(grid, outfile)

t = transformation.decomposition(grid.t)[0]
grid.t = grid.t * transformation.matrix(s = [20, 20], r=0.1)
grid.t = transformation.matrix(t = t, s = [20, -20], r=0.1)
print grid.t
storage.writeRaster(grid, outfile)