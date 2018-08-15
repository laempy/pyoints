"""In this example we learn the basics of processing point clound data using 
Pyoints. 


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


Get the spatial extent in 3D.

>>> print(np.round(lasReader.extent, 2).tolist())
[364186.82, 5509577.66, -1.7, 364196.82, 5509587.66, 29.66]


Now we actually load the point cloud data from disc. We recieve an instance of
`GeoRecords`, which is an extension of a numpy record array.

>>> las = lasReader.load()

>>> len(las)
10427269
>>> print(las.shape)
(10427269,)


Get some information on the fields.

>>> print(sorted(las.dtype.descr))
[('classification', '|u1'), ('coords', '<f8', (3,)), ('intensity', '<u2'), ('user_data', '|u1')]

>>> print(las[0:10].intensity)
[176 204 184 219 216 209 174 200 191 208]

>>> print(np.unique(las.classification))
[2 5]


Take a look at the extent in 2D and 3D

>>> print(np.round(las.extent(2), 2).tolist())
[364186.82, 5509577.66, 364196.82, 5509587.66]
>>> print(np.round(las.extent(3), 2).tolist())
[364186.82, 5509577.66, -1.7, 364196.82, 5509587.66, 29.66]


Now we take a closer look at the spatial index. We begin with selecting all 
points close to the corners of the point cloud.

>>> radius = 1.0
>>> corners = las.extent().corners
>>> print(np.round(corners).astype(int))
[[ 364187 5509578      -2]
 [ 364197 5509578      -2]
 [ 364197 5509588      -2]
 [ 364187 5509588      -2]
 [ 364187 5509588      30]
 [ 364197 5509588      30]
 [ 364197 5509578      30]
 [ 364187 5509578      30]]
    

But, before we select the points, we count the number of neighbours within the
radius. 

>>> count = las.indexKD().ball_count(radius, coords=corners)
>>> print(count)
[53494 59072 32761  5892     0     0     0     0]


OK, now we actually select the points.

>>> n_ids = las.indexKD().ball(corners, radius)
>>> print(len(n_ids))
8


For each point we recieve a list of indices. So we concatenate them to save 
the resulting subset as a point cloud.
 
>>> n_ids = np.concatenate(n_ids).astype(int)
>>> print(len(n_ids))
151219

>>> outfile = os.path.join(outpath, 'base_ball.las')
>>> storage.writeLas(las[n_ids], outfile)


We also can select the `k` nearest neighbours. 

>>> dists, n_ids = las.indexKD().knn(corners, k=2)

We recieve a matrix of distances and a matrix of indices.

>>> print(np.round(dists, 2))
[[0.25 0.25]
 [0.03 0.03]
 [0.61 0.61]
 [0.91 0.91]
 [3.95 3.98]
 [2.63 2.63]
 [3.04 3.04]
 [2.44 2.45]]
>>> print(n_ids)
[[4674322 4674113]
 [7599852 7599856]
 [9337257 9337216]
 [9453720 9454345]
 [8430539 8430732]
 [7921572 7919512]
 [8306006 3216760]
 [  41041   41045]]


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
[[ 546280 4958620      -1]
 [ 546295 4958620      -1]
 [ 546295 4958629      -1]
 [ 546280 4958629      -1]
 [ 546280 4958629      15]
 [ 546295 4958629      15]
 [ 546295 4958620      15]
 [ 546280 4958620      15]]


Finally we apply the query and save the subset.

>>> n_ids = indexKD.ball(s_corners, radius)
>>> n_ids = np.concatenate(n_ids).astype(int)
>>> print(len(n_ids))
146166

>>> outfile = os.path.join(outpath, 'base_ellipsoid.las')
>>> storage.writeLas(las[n_ids], outfile)

"""
