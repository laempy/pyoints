"""Test to read and write raster files."""

import os

from pointspy import (
    storage,
    transformation,
    projection,
)


inpath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'images')
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# load image
#############

infile = os.path.join(inpath, 'logo.png')
print("load %s" % infile)
proj = projection.Proj.from_epsg(32632)
rasterHandler = storage.RasterReader(infile, proj=proj)
grid = rasterHandler.load()
print(grid.shape)
print(grid.dtype.descr)

# apply transformation
T = transformation.matrix(
        t=[332575, 5513229], s=[0.5, -0.5], r=0.1, order='srt')
grid.transform(T)


# save image and load again
############################

outfile = os.path.join(outpath, 'test.tif')
print("save %s" % outfile)
storage.writeRaster(grid, outfile)

print("load %s" % outfile)
rasterHandler = storage.RasterReader(outfile, proj=projection.Proj())
grid = rasterHandler.load()

print(grid.t.origin)
