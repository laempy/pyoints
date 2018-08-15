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


Define input and output path.

>>> inpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), '')
>>> outpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), 'output')

Select a input LAS point cloud.

>>> infile = os.path.join(inpath, 'plot17_subset.las')
>>> lasReader = storage.LasReader(infile)
>>> las = lasReader.load()


Now let's convert the point cloud to a raster. All points within a raster cell
are grouped to an individual point cloud.

>>> T = transformation.matrix(t=las.t.origin[:2], s=[1.0, 2.0])
>>> raster = grid.voxelize(las, T)


Let's inspect the properties of the raster.

>>> print(raster.shape)
(5, 10)
>>> print(raster.dtype)
object
>>> print(len(raster[0, 0]))
187284
>>> print(sorted(raster[0, 0].dtype.descr))
[('classification', '|u1'), ('coords', '<f8', (3,)), ('intensity', '<u2'), ('user_data', '|u1')]


We can save the points of specific cells indiviudally.

>>> outfile = os.path.join(outpath, 'grid_cell.las')
>>> storage.writeLas(raster[3, 5], outfile)


We create a new raster, aggregating the point cloud data in a more specific
manner.

>>> T = transformation.matrix(t=las.t.origin[:2], s=[0.5, 0.5])
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
(20, 20)
>>> print(sorted(raster.dtype.descr))
[('cell_count', '<i8'), ('coords', '<f8', (2,)), ('z', '<f8', (3,))]


Save the fields as individual raster images.

>>> outfile = os.path.join(outpath, 'grid_count.tif')
>>> storage.writeRaster(raster, outfile, field='cell_count')

>>> outfile = os.path.join(outpath, 'grid_z.tif')
>>> storage.writeRaster(raster, outfile, field='z')


Now let's create a three dimensional voxel space.

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
(63, 25, 25)
>>> print(sorted(voxels.dtype.descr))
[('cell_count', '<i8'), ('coords', '<f8', (3,)), ('intensity', '<i8')]

Finally save only the non empty voxel cells.

>>> outfile = os.path.join(outpath, 'grid_voxels.las')
>>> storage.writeLas(voxels[voxels.cell_count > 0], outfile)

"""