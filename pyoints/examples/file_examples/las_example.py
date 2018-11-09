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
# along with Pyoints. If not, see <https://www.gnu.org/licenses/>.
# END OF LICENSE NOTE
"""Learn how to save and load LAS-files.

>>> import os
>>> import numpy as np
>>> from pyoints import storage

Create an output path.

>>> outpath = os.path.join(
...             os.path.dirname(os.path.abspath(__file__)), '..', 'output')

Create GeoRecords from scratch.

>>> geoRecords = storage.misc.create_random_GeoRecords(
...                     center=[332592.88, 5513244.80, 120], epsg=25832)
>>> print(geoRecords.shape)
(1000,)
>>> print(sorted(geoRecords.dtype.descr))
[('classification', '<u8'), ('coords', '<f8', (3,)), ('intensity', '<u8'), ('keypoint', '|b1'), ('synthetic', '|b1'), ('values', '<f8'), ('withheld', '|b1')]


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
[('classification', '|u1'), ('coords', '<f8', (3,)), ('intensity', '|u1'), ('keypoint', '|b1'), ('synthetic', '|b1'), ('values', '<f8'), ('withheld', '|b1')]
    
Check values.

>>> np.all(geoRecords.classification == las.classification)
True
>>> np.all(geoRecords.synthetic == las.synthetic)
True
>>> np.all(geoRecords.keypoint == las.keypoint)
True
>>> np.all(geoRecords.withheld == las.withheld)
True
>>> np.all(geoRecords.intensity == las.intensity)
True

"""
