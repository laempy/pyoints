"""In this example we learn the basics of point cloud processing using Pyoints. 


We begin with loading the required modules.

>>> import os
>>> import numpy as np

>>> from pyoints import (
...     storage,
...     transformation,
...     IndexKD,
... )


Define input and output path.

>>> inpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), 'data')
>>> outpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), 'output')


Select a input LAS point cloud.

>>> infile = os.path.join(inpath, 'forest.las')
>>> lasReader = storage.LasReader(infile)


Get the origin of the point cloud.

>>> print(np.round(lasReader.t.origin, 2).tolist())
[364187.98, 5509577.71, -1.58]


Get the projection of the point cloud.

>>> print(lasReader.proj.proj4)
+proj=utm +zone=32 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs


Get the number of points.

>>> print(len(lasReader))
482981


Get the spatial extent in 3D.

>>> print(np.round(lasReader.extent, 2).tolist())
[364187.98, 5509577.71, -1.58, 364194.95, 5509584.71, 28.78]


Now we actually load the point cloud data from disc. We recieve an instance of
`GeoRecords`, which is an extension of a numpy record array.

>>> las = lasReader.load()

>>> print(las.shape)
(482981,)


Get some information on the fields.

>>> print(sorted(las.dtype.descr))
[('classification', '|u1'), ('coords', '<f8', (3,)), ('intensity', '<u2')]

>>> print(las[0:10].intensity)
[216 213 214 199 214 183 198 209 200 199]

>>> print(np.unique(las.classification))
[2 5]


Take a look at the extent in 2D and 3D

>>> print(np.round(las.extent(2), 2).tolist())
[364187.98, 5509577.71, 364194.95, 5509584.71]
>>> print(np.round(las.extent(3), 2).tolist())
[364187.98, 5509577.71, -1.58, 364194.95, 5509584.71, 28.78]


Now we take a closer look at the spatial index. We begin with selecting all 
points close to the corners of the point cloud.

>>> radius = 1.0
>>> corners = las.extent().corners
>>> print(np.round(corners).astype(int))
[[ 364188 5509578      -2]
 [ 364195 5509578      -2]
 [ 364195 5509585      -2]
 [ 364188 5509585      -2]
 [ 364188 5509585      29]
 [ 364195 5509585      29]
 [ 364195 5509578      29]
 [ 364188 5509578      29]]
    

But, before we select the points, we count the number of neighbours within the
radius. 

>>> count = las.indexKD().ball_count(radius, coords=corners)
>>> print(count)
[2502 1984 4027  475    0    0    0    0]


OK, now we actually select the points.

>>> n_ids = las.indexKD().ball(corners, radius)
>>> print(len(n_ids))
8


For each point we recieve a list of indices. So we concatenate them to save 
the resulting subset as a point cloud.
 
>>> n_ids = np.concatenate(n_ids).astype(int)
>>> print(len(n_ids))
8988

>>> outfile = os.path.join(outpath, 'base_ball.las')
>>> storage.writeLas(las[n_ids], outfile)


We also can select the `k` nearest neighbours. 

>>> dists, n_ids = las.indexKD().knn(corners, k=2)

We recieve a matrix of distances and a matrix of indices.

>>> print(np.round(dists, 2))
[[0.3  0.3 ]
 [0.02 0.04]
 [0.49 0.49]
 [0.62 0.62]
 [3.95 3.97]
 [5.71 5.73]
 [1.26 1.27]
 [1.65 1.66]]
>>> print(n_ids)
[[     6  16742]
 [ 92767  92763]
 [320695 321128]
 [206255 206239]
 [440696 440687]
 [400070 400050]
 [400369 400340]
 [365239 365240]]


Again, we save the resulting subset.

>>> n_ids = n_ids.flatten()
>>> len(n_ids)
16

>>> outfile = os.path.join(outpath, 'base_knn.las')
>>> storage.writeLas(las[n_ids], outfile)


If we need to select points in the shape of an ellipsoid, we also can create a 
scaled spatial index. Doing so, each coordinate axis is scaled individually.

>>> T = transformation.s_matrix([1.5, 0.9, 0.5])
>>> indexKD = IndexKD(las.coords, T=T)


Again, we select neighbouring points using the `ball` query. But here, we need
to scale the input coordinates before.

>>> s_corners = T.to_local(corners)
>>> print(np.round(s_corners).astype(int))
[[ 546282 4958620      -1]
 [ 546292 4958620      -1]
 [ 546292 4958626      -1]
 [ 546282 4958626      -1]
 [ 546282 4958626      14]
 [ 546292 4958626      14]
 [ 546292 4958620      14]
 [ 546282 4958620      14]]


Finally we apply the query and save the subset.

>>> n_ids = indexKD.ball(s_corners, radius)
>>> n_ids = np.concatenate(n_ids).astype(int)
>>> print(len(n_ids))
8520

>>> outfile = os.path.join(outpath, 'base_ellipsoid.las')
>>> storage.writeLas(las[n_ids], outfile)

"""
