"""In this tutorial we try to dectect stems in a forest using a point cloud
of terrestrial laser scans. A couple of .las-files will be generated, which
should be inspected with software like CloudCompare.



Let's load the required modules.

>>> import os
>>> import numpy as np
>>> from pyoints import (
...     storage,
...     filters,
...     clustering,
...     classification,
...     vector,
...     Surface,
...     GeoRecords,
...     interpolate,
... )


First, we define an input and an output path.

>>> inpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), '')
>>> outpath = os.path.join(
...                 os.path.dirname(os.path.abspath(__file__)), 'output')


After that, we load a input LAS point cloud.

>>> infile = os.path.join(inpath, 'plot17_subset.las')
>>> lasReader = storage.LasReader(infile)
>>> las = lasReader.load()
>>> print(len(las))
10427269


We begin with deriving a digital elevation model to calculate the height above 
ground. So we load all points classified as ground `grd`.

>>> grd = las.grd()
>>> f_ids = filters.dem_filter(grd.coords, 0.5)
>>> grd = grd[f_ids]
>>> print(len(grd))
299

>>> dem = Surface(grd.coords)


We save the ground points in a seperate .las-file.

>>> outfile = os.path.join(outpath, 'stemfilter_dem.las')
>>> storage.writeLas(grd, outfile)


For the stem detection we will focus on points with height above ground greater
0.5 m.

>>> height = las.coords[:, 2] - dem(las.coords)
>>> s_ids = np.where(height > 0.5)[0]
>>> print(len(s_ids))
4967153


We filter the point cloud using a small filter radius. Only a subset of points
is kept, with a point distance of at least 10 cm.

>>> f_ids = list(filters.ball(las.indexKD(), 0.1, order=s_ids))
>>> las = las[f_ids]
>>> print(len(las))
59931

>>> outfile = os.path.join(outpath, 'stemfilter_ball_10.las')
>>> storage.writeLas(las, outfile)


We only keep points with a lot of neighbours to reduce noise. 

>>> count = las.indexKD().ball_count(0.3)
>>> mask = count > 20
>>> las = las[mask]
>>> print(len(las))
37210

>>> outfile = os.path.join(outpath, 'stemfilter_denoised.las')
>>> storage.writeLas(las, outfile)


Now, we will filter with a radius of 1 m. This results in a point cloud
with point distances of at least 1 m. Here points associated with stems are 
arranged in straight lines.

>>> f_ids = list(filters.ball(las.indexKD(), 1.0))
>>> las = las[f_ids]
>>> print(len(las))
440

>>> outfile = os.path.join(outpath, 'stemfilter_ball_100.las')
>>> storage.writeLas(las, outfile)

For dense point clouds the filtering technique results in point distances 
between 1 m and 2 m. So, we can assume that linear arranged points should
have 2 to 3 neighbouring points within a radius of 1.5 m.

>>> count = las.indexKD().ball_count(1.5)
>>> mask = np.all((count >= 2, count <= 3), axis=0)
>>> las = las[mask]
>>> print(len(las))
134

>>> outfile = os.path.join(outpath, 'stemfilter_linear.las')
>>> storage.writeLas(las, outfile)


Now the stems are clearly visible in the point cloud. So, we can detect the 
stems by clustering the points.

>>> cluster_indices = clustering.dbscan(las.indexKD(), 2, epsilon=1.5)

>>> print(len(cluster_indices))
134
>>> print(np.unique(cluster_indices))
[-1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]


In the next step we remove small clusters and unassigned points.

>>> cluster_dict = classification.classes_to_dict(cluster_indices, min_size=6)
>>> cluster_indices = classification.dict_to_classes(cluster_dict, len(las))

>>> print(list(cluster_dict.keys()))
[15, 3, 4, 13, 7]


We add an additional field to the point cloud to store the tree number.

>>> las = las.add_fields([('tree_id', int)], data=[cluster_indices])

>>> outfile = os.path.join(outpath, 'stemfilter_stems.las')
>>> storage.writeLas(las, outfile)


Now we can fit a vector to each stem. You should take a close look at the 
characteristics of the `Vector` object.

>>> stems = {}
>>> for tree_id in cluster_dict:
...     coords = las[cluster_dict[tree_id]].coords
...     stem = vector.Vector.from_coords(coords)
...     stems[tree_id] = stem


Finally we determinate the tree root coordinates using the previously derived
digital elevation model.

>>> roots = []
>>> for tree_id in stems:
...     coord = stems[tree_id].surface_intersection(dem)
...     roots.append((tree_id, coord))

>>> dtype = [('tree_id', int), ('coords', float, 3)]
>>> roots = np.array(roots, dtype=dtype).view(np.recarray)
>>> roots = GeoRecords(las.proj, roots)

>>> outfile = os.path.join(outpath, 'stemfilter_roots.las')
>>> storage.writeLas(roots, outfile)

"""
