# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive licencing 
# model will be made before releasing this software.
# END OF LICENSE NOTE
"""Learn how to save and load LAS-files.

>>> import os
>>> import numpy as np
>>> from pyoints import storage

>>> outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

Create GeoRecords from scratch

>>> geoRecords = storage.misc.create_random_GeoRecords(
...                     center=[332592.88, 5513244.80, 120], epsg=25832)
>>> print(geoRecords.shape)
(1000,)
>>> print(sorted(geoRecords.dtype.descr))
[('classification', '<i8'), ('coords', '<f8', (3,)), ('intensity', '<i8'), ('values', '<f8')]

Save as LAS-file.

>>> outfile = os.path.join(outpath, 'test.las')
>>> storage.writeLas(geoRecords, outfile)

Load LAS-file again and check the characteristics.

>>> lasReader = storage.LasReader(outfile)
>>> print(len(lasReader))
1000

>>> las = lasReader.load()
>>> print(las.shape)
(1000,)
>>> print(sorted(las.dtype.descr))
[('classification', '|u1'), ('coords', '<f8', (3,)), ('intensity', '<u2'), ('values', '<f8')]

"""