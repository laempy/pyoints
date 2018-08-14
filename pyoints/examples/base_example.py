"""Learn how to process point clound data using Pyoints.

In this tutorial we load a .las point cloud of a forrest to extract stems. 

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

Get the origin of the point cloud.

>>> print(np.round(lasReader.t.origin, 2).tolist())
[364186.82, 5509577.66, -1.7]

Get the projection of the point cloud.

>>> print(lasReader.proj.proj4)
+proj=utm +zone=32 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs

Get the number of points.

>>> print(len(lasReader))
10427269

Get the spatial extent in 2D.

>>> print(np.round(lasReader.extent, 2).tolist())
[364186.82, 5509577.66, -1.7, 364196.82, 5509587.66, 29.66]

Now we actually load the point cloud data from disc. We recieve an instance of
`GeoRecords`, which is an extension of numpy recarray.

>>> las = lasReader.load()
>>> len(las)
10427269
>>> print(las.shape)
(10427269,)
>>> print(np.round(las.extent(2), 2).tolist())
[364186.82, 5509577.66, 364196.82, 5509587.66]
>>> print(np.round(las.extent(3), 2).tolist())
[364186.82, 5509577.66, -1.7, 364196.82, 5509587.66, 29.66]

Get some information on the fields.

>>> print(sorted(las.dtype.names))
['classification', 'coords', 'intensity', 'user_data']

>>> print(las[0:10].intensity)
[176 204 184 219 216 209 174 200 191 208]

>>> print(np.unique(las.classification))
[2 5]

Now let's convert the point cloud to a raster. All points within a raster cell
are grouped.

>>> T = transformation.matrix(t=las.t.origin[:2], s=[1.0, 2.0])
>>> raster = grid.voxelize(las, T)

>>> print(raster.shape)
(5, 10)
>>> print(raster.dtype)
object
>>> print(len(raster[0, 0]))
187284
>>> print(sorted(raster[0, 0].dtype.descr))
[('classification', '|u1'), ('coords', '<f8', (3,)), ('intensity', '<u2'), ('user_data', '|u1')]

Let's save the points of a specific cell.

>>> outfile = os.path.join(outpath, 'raster_cell.las')
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
...     return (n, [z_min, z_mean, z_max], z_min)

>>> dtype=[('cell_count', int), ('z', float, 3), ('dem', float)]
>>> raster = grid.voxelize(las, T, agg_func=aggregate_function, dtype=dtype)
>>> raster = Grid(las.proj, raster, T)

>>> print(raster.shape)
(20, 20)
>>> print(sorted(raster.dtype.descr))
[('cell_count', '<i8'), ('coords', '<f8', (2,)), ('dem', '<f8'), ('z', '<f8', (3,))]


Save the fields as individual raster images.

>>> outfile = os.path.join(outpath, 'raster_count.tif')
>>> storage.writeRaster(raster, outfile, field='cell_count')

>>> outfile = os.path.join(outpath, 'raster_z.tif')
>>> storage.writeRaster(raster, outfile, field='z')

>>> outfile = os.path.join(outpath, 'raster_dem.tif')
>>> storage.writeRaster(raster, outfile, field='dem')


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

>>> outfile = os.path.join(outpath, 'voxels.las')
>>> storage.writeLas(voxels[voxels.cell_count > 0], outfile)


Now let's filter ground surface points using the original point cloud.

>>> filter_radius = 0.3
>>> f_ids = filters.dem_filter(las.coords, filter_radius)

>>> print(len(f_ids))
806
>>> print(f_ids[0:5])
[6672731 7604578 7423139 7474770 7496207]

>>> outfile = os.path.join(outpath, 'dem.las')
>>> storage.writeLas(las[f_ids], outfile)


We can use these surface points to create a surface interpolator, which 
calculates the height above ground.

>>> interpolator = Surface(las.coords[f_ids, :])

We can interact between a grid and a point cloud by converting the grid to a
point cloud using `records()`. So, we update voxel coordinates using the 
surface interpolator.

>>> ground_voxels = voxels[0, :, :]
>>> height = interpolator(ground_voxels.records().coords)
>>> ground_voxels.records()['coords'][:, 2] = height

>>> outfile = os.path.join(outpath, 'ground_voxels.las')
>>> storage.writeLas(ground_voxels, outfile)



"""
