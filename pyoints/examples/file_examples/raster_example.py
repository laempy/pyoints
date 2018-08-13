# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive licencing 
# model will be made before releasing this software.
# END OF LICENSE NOTE
"""Learn how to save and load image files.

>>> import os
>>> from pyoints import (
...     storage,
...     transformation,
...     projection,
... )

Create input and output path.

>>> inpath = os.path.join(
...     os.path.dirname(os.path.abspath(__file__)), 'data')
>>> outpath = os.path.join(
...     os.path.dirname(os.path.abspath(__file__)), 'output')

Load an image file.

>>> infile = os.path.join(inpath, 'logo_pyoints.jpg')
>>> proj = projection.Proj.from_epsg(32632)
>>> rasterHandler = storage.RasterReader(infile, proj=proj)
>>> raster = rasterHandler.load()

>>> print(raster.shape)
(96, 250)
>>> print(sorted(raster.dtype.descr))
[('bands', '<i8', (3,)), ('coords', '<f8', (2,))]

Apply a transformation to the matrix to get a propper spatial reference.

>>> T = transformation.matrix(
...         t=[332575, 5513229], s=[0.5, -0.5], r=0.1, order='srt')
>>> raster.transform(T)

Save the image as an tif-file. You might like to check the spatial reference of
the output image using a Geographic Information System (GIS).

>>> outfile = os.path.join(outpath, 'test.tif')
>>> storage.writeRaster(raster, outfile)

Load image again and check characteristics.

>>> rasterHandler = storage.RasterReader(outfile, proj=projection.Proj())
>>> print(rasterHandler.t.origin)
[ 332575. 5513229.]

>>> raster = rasterHandler.load()
>>> print(raster.t.origin)
[ 332575. 5513229.]
>>> print(raster.shape)
 (96, 250)
 >>> print(sorted(raster.dtype.descr))
[('bands', '<i8', (3,)), ('coords', '<f8', (2,))]

"""