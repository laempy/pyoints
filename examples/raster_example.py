# BEGIN OF LICENSE NOTE
# This file is part of PoYnts.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
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
raster = rasterHandler.load()
print(raster.shape)
print(raster.dtype.descr)

# apply transformation
T = transformation.matrix(
        t=[332575, 5513229], s=[0.5, -0.5], r=0.1, order='srt')
raster.transform(T)


# save image and load again
############################

outfile = os.path.join(outpath, 'test.tif')
print("save %s" % outfile)
storage.writeRaster(raster, outfile)

print("load %s" % outfile)
rasterHandler = storage.RasterReader(outfile, proj=projection.Proj())
raster = rasterHandler.load()

print(raster.t.origin)
