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
"""In this example we load a point cloud and convert it to rasters and voxels.



>>> import os
>>> import numpy as np

>>> from pyoints import (
...     storage,
...     projection,
...     transformation,
...     grid,
...     Grid,
...     filters,
...     Surface,
... )


First, we define input and output paths.

>>> inpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), 'data')
>>> outpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), 'output')


Then, we select an input LAS point cloud.

>>> infile = os.path.join(inpath, 'forest.las')
>>> lasReader = storage.LasReader(infile)
>>> las = lasReader.load()


Now, let's convert the point cloud to a raster. All points within a raster cell
are grouped to an individual point cloud.

>>> T = transformation.matrix(t=las.t.origin[:2], s=[1.0, 2.0])
>>> raster = grid.voxelize(las, T)


Let's inspect the properties of the raster.

>>> print(raster.shape)
(4, 7)
>>> print(raster.dtype)
object
>>> print(len(raster[0, 0]))
11473
>>> print(sorted(raster[0, 0].dtype.descr))
[('classification', '|u1'), ('coords', '<f8', (3,)), ('intensity', '<u2')]


We can save the points of specific cells individually.

>>> outfile = os.path.join(outpath, 'grid_cell.las')
>>> storage.writeLas(raster[2, 1], outfile)


We create a new raster, aggregating the point cloud data in a more specific
manner.

>>> T = transformation.matrix(t=las.t.origin[:2], s=[0.3, 0.3])
>>> def aggregate_function(ids):
...     z = las.coords[ids, 2]
...     n = len(ids)
...     z_min = z.min() if n > 0 else 0
...     z_mean = z.mean() if n > 0 else 0
...     z_max = z.max() if n > 0 else 0
...     return (n, [z_min, z_mean, z_max])

>>> dtype=[('cell_count', int), ('z', float, 3)]
>>> raster = grid.voxelize(las, T, agg_func=aggregate_function, dtype=dtype)
>>> raster = Grid(las.proj, raster, T)

>>> print(raster.shape)
(24, 24)
>>> print(sorted(raster.dtype.descr))
[('cell_count', '<i8'), ('coords', '<f8', (2,)), ('z', '<f8', (3,))]


We save the fields as individual raster images.

>>> outfile = os.path.join(outpath, 'grid_count.tif')
>>> storage.writeRaster(raster, outfile, field='cell_count')

>>> outfile = os.path.join(outpath, 'grid_z.tif')
>>> storage.writeRaster(raster, outfile, field='z')


Now, let's create a three dimensional voxel space.

>>> T = transformation.matrix(t=las.t.origin, s=[0.4, 0.4, 0.5])
>>> def aggregate_function(ids):
...     intensity = las.intensity[ids]
...     n = len(ids)
...     intensity = intensity.mean() if n > 0 else 0
...     return (n, intensity)

>>> dtype=[('cell_count', int), ('intensity', int)]

>>> voxels = grid.voxelize(las, T, agg_func=aggregate_function, dtype=dtype)
>>> voxels = Grid(las.proj, voxels, T)

>>> print(voxels.shape)
(61, 18, 18)
>>> print(sorted(voxels.dtype.descr))
[('cell_count', '<i8'), ('coords', '<f8', (3,)), ('intensity', '<i8')]


Finally, we save only the non-empty voxel cells.

>>> outfile = os.path.join(outpath, 'grid_voxels.las')
>>> storage.writeLas(voxels[voxels.cell_count > 0], outfile)

"""
