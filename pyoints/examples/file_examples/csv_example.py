# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive licencing 
# model will be made before releasing this software.
# END OF LICENSE NOTE
"""Learn how to save and load .csv-files.

>>> import os
>>> from pyoints import storage

Create an output path relative to the source file.

>>> outpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), 'output')

Create GeoRecords from scratch.

>>> geoRecords = storage.misc.create_random_GeoRecords(
...                     center=[332592.88, 5513244.80, 120], epsg=25832)

>>> print(geoRecords.shape)
(1000,)
>>> print(sorted(geoRecords.dtype.descr))
[('classification', '<i8'), ('coords', '<f8', (3,)), ('intensity', '<i8'), ('values', '<f8')]

Save as .csv-file.

>>> outfile = os.path.join(outpath, 'test.csv')
>>> storage.writeCsv(geoRecords, outfile)

Load .csv-file again and check the characteristics.

>>> geoRecords = storage.loadCsv(outfile, header=True)
>>> print(geoRecords.shape)
(1000,)
>>> print(sorted(geoRecords.dtype.descr))
[('classification', '<f8'), ('coords', '<f8', (3,)), ('index', '<i8'), ('intensity', '<f8'), ('values', '<f8')]

"""
