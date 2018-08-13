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
# END OF LICENSE NOTE
"""Learn how to save and load DUMP-files.

>>> import os
>>> import numpy as np
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

Save as DUMP-file.

>>> outfile = os.path.join(outpath, 'test.pydump')
>>> storage.writeDump(geoRecords, outfile)

Load DUMP-file again and check the characteristics.

>>> dumpReader = storage.DumpReader(outfile)
>>> geoRecords = dumpReader.load()

>>> print(geoRecords.shape)
(1000,)
>>> print(sorted(geoRecords.dtype.descr))
[('classification', '<i8'), ('coords', '<f8', (3,)), ('intensity', '<i8'), ('values', '<f8')]
    
"""
