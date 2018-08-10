# BEGIN OF LICENSE NOTE
# This file is part of Pyoints.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive licencing 
# model will be made before releasing this software.
# END OF LICENSE NOTE
"""Convertion of grid data structures.
"""

import pandas
import numpy as np

from .. import (
    assertion,
    transformation,
    registration,
    Extent,
)

from .transformation import (
    coords_to_keys,
    keys_to_indices,
    indices_to_keys,
)


def voxelize(rec, T, shape=None, agg_func=None, dtype=None):
    """Aggregate a point cloud to a voxel or raster.

    Parameters
    ----------
    rec : np.recarray(shape=(n, ))
        Numpy record array of `n` points to voxelize. It requires a two
        dimensional field 'coords' associated with `k` dimensional coordinates.
    T : array_like(Number, shape=(m+1, m+1))
        Transformation matrix defining the `m <= k` dimensional voxel system.
    shape : optional, array_like(int, shape=(m))
        Shape of the output voxel space. If None, `shape` is fit to
        `rec.coords`.
    agg_func : optional, callable
        Function to aggregate the record array. If None, `agg_func` set to
        `lambda ids: rec[ids]`.
    dtype : optional, np.dtype
        Output data type. If None, set to `[type(rec)]`.

    Returns
    -------
    np.recarray(dtype=dtype)
        Output `m` dimensional matrix.

    See Also
    --------
    Grid

    Examples
    --------

    Create record array with coordinates.

    >>> coords = [(0, 0), (1, 0.5), (2, 2), (4, 6), (3, 2), (1, 5), (3, 0)]
    >>> rec = np.recarray(len(coords), dtype=[('coords', float, 2)])
    >>> rec['coords'] = coords

    Voxelize records.

    >>> T = transformation.matrix(s=(2.5, 3), t=[0, 0])
    >>> grid = voxelize(rec, T)

    >>> print(grid.dtype)
    object
    >>> print(grid.shape)
    (3, 2)
    >>> print(grid[0, 0].coords)
    [[0.  0. ]
     [1.  0.5]
     [2.  2. ]]

    Voxelize with aggregation function.

    >>> dtype = [('points', type(rec)), ('count', int)]
    >>> agg_func = lambda ids: (rec[ids], len(ids))
    >>> grid = voxelize(rec, T, agg_func=agg_func, dtype=dtype)

    >>> print(grid.dtype.descr)
    [('points', '|O'), ('count', '<i8')]
    >>> print(grid.shape)
    (3, 2)
    >>> print(grid.count)
    [[3 2]
     [1 0]
     [0 1]]
    >>> print(grid[0, 0].points.coords)
    [[0.  0. ]
     [1.  0.5]
     [2.  2. ]]

    Voxelize three dimensional coordinates to recieve a two dimensional raster.

    >>> coords = [(0, 0, 1), (-2, 0.3, 5), (2, 2, 3), (4, 6, 2), (3, 2, 1)]
    >>> rec = np.recarray(len(coords), dtype=[('coords', float, 3)])
    >>> rec['coords'] = coords

    >>> T = transformation.matrix(s=(2.5, 3), t=[0, 0])
    >>> grid = voxelize(rec, T)
    >>> print(grid.shape)
    (3, 2)
    >>> print(grid[0, 0].coords)
    [[0. 0. 1.]
     [2. 2. 3.]]

    """
    if not isinstance(rec, np.recarray):
        raise TypeError("'rec' need an instance of np.recarray")
    if 'coords' not in rec.dtype.names:
        raise ValueError("'rec' requires field 'coords'")

    if agg_func is None:
        def agg_func(ids): return rec[ids]
    elif not callable(agg_func):
        raise ValueError("'agg_func' needs to be callable")

    coords = assertion.ensure_coords(rec.coords, min_dim=T.dim)
    keys = coords_to_keys(T, coords[:, :T.dim])

    if shape is None:
        shape = tuple(keys.max(0)[:T.dim] + 1)
    else:
        shape = assertion.ensure_shape(shape, dim=T.dim)

    # group keys
    df = pandas.DataFrame({'indices': keys_to_indices(keys, shape)})
    groupDict = df.groupby(by=df.indices).groups
    keys = indices_to_keys(list(groupDict.keys()), shape)

    # cut by extent
    min_mask = np.all(keys >= 0, axis=1)
    max_mask = np.all(keys < shape, axis=1)
    mask = np.all((min_mask, max_mask), axis=1)

    # create lookup array
    lookup = np.empty(shape, dtype=list)
    lookup.fill([])
    lookup[tuple(keys.T[mask].tolist())] = list(groupDict.values())

    # Aggregate per cell
    if dtype is None:
        otypes = [type(rec)]
        v = np.vectorize(agg_func, otypes=otypes)
        res = v(lookup)
    else:
        dtype = np.dtype(dtype)
        otypes = [dtype[name].str for name in dtype.names]
        v = np.vectorize(agg_func, otypes=otypes)
        res = np.recarray(lookup.shape, dtype=dtype)
        v_arrays = v(lookup)
        for i, name in enumerate(dtype.names):
            res[name][:] = v_arrays[i]
    return res


def corners_to_transform(corners, scale=None):
    """Create a transformation matrix using the corners of a raster.

    Parameters
    ----------
    corners : array_like(Number, shape=(2\*\*k, k))
        Corners of a `k` dimensional grid in a `k` dimensional space.
    scale : optional, array_like(Number, shape=(k))
        Optional scale to define the pixel resolution of a raster.

    Examples
    --------

    Create some corners.

    >>> T = transformation.matrix(t=[3, 5], s=[10, 20], r=np.pi/2)
    >>> coords = Extent([np.zeros(2), np.ones(2)]).corners
    >>> corners = transformation.transform(coords, T)

    Create transformation matrix without scale.

    >>> M = corners_to_transform(corners)
    >>> print(np.round(M, 3))
    [[ 0. -1.  3.]
     [ 1.  0.  5.]
     [ 0.  0.  1.]]

    Create a transformation matrix with a scale.

    >>> M = corners_to_transform(corners, [0.5, 2])
    >>> print(np.round(M, 3))
    [[ 0.  -2.   3. ]
     [ 0.5  0.   5. ]
     [ 0.   0.   1. ]]

    """
    corners = assertion.ensure_coords(corners)
    dim = corners.shape[1]
    pts = Extent([np.zeros(dim), np.ones(dim)]).corners

    # find transformation matrix
    T = registration.find_transformation(corners, pts)

    # get translation, rotation and scale
    t, r, s, det = transformation.decomposition(T)

    return transformation.matrix(t=t, r=r, s=scale)


def transform_to_corners(T, shape):
    """Generates the corners of a grid using a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix in a `k` dimensional space.
    shape : array_like(int, shape=(k))
        Desired shape of the grid.

    Returns
    -------
    np.ndarray(Number, shape=(2\*\*k, k))
        Desired corners of the grid.

    Examples
    --------

    >>> T = transformation.matrix(t=[10, 20], s=[0.5, 2])
    >>> print(np.round(T, 3))
    [[ 0.5  0.  10. ]
     [ 0.   2.  20. ]
     [ 0.   0.   1. ]]

    >>> corners = transform_to_corners(T, (100, 200))
    >>> print(np.round(corners, 3))
    [[ 10.  20.]
     [ 60.  20.]
     [ 60. 420.]
     [ 10. 420.]]

    """
    ext = Extent([np.zeros(len(shape)), shape])
    return transformation.transform(ext.corners, T)
