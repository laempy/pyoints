"""Learn how to process point clound data using Pyoints.

In this tutorial we load a .las point cloud of a forrest to extract stems. 

>>> import os
>>> import numpy as np

>>> from pyoints import (
...     storage,
...     projection,
...     transformation,
...     grid,
... )

Select input file.

>>> inpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), '')
>>> infile = os.path.join(inpath, 'plot17_subset.las')


Create a .las-reader to load the file.

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

>>> print(las.dtype.names)
('coords', 'intensity', 'classification', 'user_data')

>>> print(las[0])
([3.64186831e+05, 5.50957767e+06, 1.26611304e+01], 176, 5, 0)

Now let's convert the point cloud to a raster file. Storing the 

>>> T = transformation.matrix(t=las.t.origin[:2], s=[0.5, 0.5], order='rst')
>>> def aggregate_function(ids):
...     coords = las.coords[ids, :]
...     return (coords[:, 2].min(), coords[:, 2].mean(), coords[:, 2].max())
>>> dtype=[('min_z', np.int), ('mean_z', np.int), ('max_z', np.int)]
>>> dtype=[('bands', float, 3)]
>>> #raster = grid.voxelize(las, T, agg_func=aggregate_function)
>>> #print(raster)

>>> 



"""
