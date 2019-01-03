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
"""Conversion of coordinates to cell indices and reverse by applying
transformations.
"""

import numpy as np

from .. import (
    assertion,
    transformation,
    registration,
    Extent,
)


def keys_to_indices(keys, shape):
    """Converts matrix keys to indices.

    Parameters
    ----------
    keys : array_like(int)
        Keys of matrix.
    shape : array_like(int, shape=(k))
        Shape of the input matrix.

    Returns
    -------
    np.ndarray(int)
        Desired indices vector associated with the requested keys.

    """
    w = np.concatenate((
        np.array([np.product(shape[i:]) for i in range(len(shape))])[1:],
        [1]
    ))
    return (keys * w).sum(1)


def indices_to_keys(indices, shape):
    """Converts indices vector to keys of a matrix.

    Parameters
    ----------
    indices : array_like(int, shape=(n))
        Index vector to convert to matrix keys. Each element `i` specifies the
        `i`-th`element in the matrix.
    shape : array_like(int, shape=(k))
        Shape of output matrix.

    Returns
    -------
    np.ndarray(int, shape=shape)
        Desired matrix keys associated with requested indices.

    """
    indices = assertion.ensure_numvector(indices)
    shape = assertion.ensure_numvector(shape)

    keys = []
    w = np.concatenate((
        np.array([np.product(shape[i:]) for i in range(len(shape))])[1:],
        [1]
    ))
    for d in w:
        keys.append(indices / d)
        indices = indices % d
    keys = np.array(keys, dtype=int).T

    return keys


def coords_to_keys(T, coords):
    """Transforms coordinates to matrix indices.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1,k+1))
        A linear transformation matrix to transform the coordinates.
        The translation represents the origin, the rotation represents the
        orientation and the scale represents the pixel size of a raster.
    coords : array_like(Number, shape=(n, k))
        Coordinates with at least `k` dimensions to convert to indices.

    Returns
    -------
    keys : np.ndarray(int, shape=(n, k))
        Indices of the coordinates within the grid.

    See Also
    --------
    keys_to_coords, coords_to_coords

    """
    coords = assertion.ensure_numarray(coords)
    dim = coords.shape[-1]
    T = transformation.LocalSystem(assertion.ensure_tmatrix(T, dim=dim))

    s = np.product(coords.shape) // dim

    flat_coords = coords.reshape((s, dim))
    values = T.to_global(flat_coords)

    keys = values.astype(int)[:, ::-1]
    return keys.reshape(coords.shape)


def keys_to_coords(T, keys):
    """Converts indices of raster cells to coordinates.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1,k+1))
        The transformation matrix of a `k` dimensional raster.
    keys : array_like(int, shape=(n, k))
        Indices of `n` raster cells.

    Returns
    -------
    coords : array_like(Number, shape=(n, k))
        Desired coordinates of the raster cells.

    See Also
    --------
    coords_to_keys, coords_to_coords

    """
    keys = assertion.ensure_numarray(keys)
    dim = keys.shape[-1]
    T = transformation.LocalSystem(assertion.ensure_tmatrix(T, dim=dim))
    s = np.product(keys.shape) // dim
    flat_keys = keys.reshape((s, dim))[:, ::-1] + 0.5
    coords = T.to_local(flat_keys).astype(float)

    return coords.reshape(keys.shape)


def coords_to_coords(T, coords):
    """Aligns coordinates with a raster grid.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        The transformation matrix of a `k` dimensional raster.
    coords : array_like(Number, shape=(n, k))
        Coordinates to align with a raster grid.

    Returns
    -------
    coords : array_like(Number, shape=(n, k))
        Desired coordinates aligned with the grid.

    See Also
    --------
    coords_to_keys, keys_to_coords

    """
    return keys_to_coords(T, coords_to_keys(T, coords))


def corners_to_transform(corners, scale=None):
    """Creates a transformation matrix using the corners of a raster.

    Parameters
    ----------
    corners : array_like(Number, shape=(2**k, k))
        Corners of a `k` dimensional grid in a `k` dimensional space.
    scale : optional, array_like(Number, shape=(k))
        Optional scale to define the pixel resolution of a raster.

    Returns
    -------
    T : np.matrix(Number, shape=(k+1, k+1))
        Desired transformation matrix.
    shape : np.ndarray(int, shape=(k))
        Shape of a grid required to cover all corners.

    Examples
    --------

    Create some corners.

    >>> T = transformation.matrix(t=[3, 5], s=[10, 20], r=np.pi/2)
    >>> coords = Extent([np.zeros(2), np.ones(2)]).corners
    >>> corners = transformation.transform(coords, T)
    >>> print(corners)
    [[  3.   5.]
     [  3.  15.]
     [-17.  15.]
     [-17.   5.]]

    Create transformation matrix without scale.

    >>> M, shape = corners_to_transform(corners)
    >>> print(np.round(M, 3))
    [[ 0. -1.  3.]
     [ 1.  0.  5.]
     [ 0.  0.  1.]]
    >>> print(shape)
    [20 10]

    Create transformation matrix with a scale.

    >>> M, shape = corners_to_transform(corners, [0.5, 2])
    >>> print(np.round(M, 3))
    [[ 0.  -2.   3. ]
     [ 0.5  0.   5. ]
     [ 0.   0.   1. ]]
    >>> print(shape)
    [10 20]

    """
    corners = assertion.ensure_coords(corners)
    dim = corners.shape[1]

    if scale is None:
        scale = np.ones(dim)
    else:
        scale = assertion.ensure_numvector(scale, length=dim)

    # find transformation matrix
    pts = Extent([np.zeros(dim), np.sign(scale)]).corners

    T = registration.find_transformation(corners, pts)

    # get translation, rotation and scale
    t, r, s, det = transformation.decomposition(T)

    T = transformation.matrix(t=t, r=r, s=scale)
    indices = np.round(T.to_global(corners))[:, ::-1].astype(int)

    assert np.all(indices >= 0), (
        'not all indices greater zero, got %s' % str(indices))

    shape = indices.max(0).astype(int)
    return T, shape


def transform_to_corners(T, shape):
    """Generates the corners of a grid based on a transformation matrix.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1, k+1))
        Transformation matrix in a `k` dimensional space.
    shape : array_like(int, shape=(k))
        Desired shape of the grid


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


def extentinfo(T, extent):
    """Receives information on a raster subset with given boundaries.

    Parameters
    ----------
    T : array_like(Number, shape=(k+1,k+1))
        The transformation matrix of a `k` dimensional grid.
    extent : array_like(Number, shape=(2 * k))
        Desired extent.

    Returns
    -------
    T : array_like(Number, shape=(k+1,k+1))
        Extent of the original raster.
    origin_key : array_like(int, shape=(k))
        Key or index of the origin of the new transformation matrix.
    shape : np.array(int, shape=(k))
        Shape of the grid subset.

    Examples
    --------

    >>> T = transformation.matrix(t=[100, 200], s=[2, -2])
    >>> ext = Extent([150, 250, 200, 300])
    >>> M, min_corner_key, shape = extentinfo(T, ext)
    >>> print(M)
    [[  2.   0. 150.]
     [  0.  -2. 300.]
     [  0.   0.   1.]]
    >>> print(min_corner_key)
    [-50  25]
    >>> print(shape)
    [26 26]

    """
    # ensure extent
    extent = Extent(extent)
    T = assertion.ensure_tmatrix(T)

    dim = len(extent) // 2

    if not T.shape[0] - 1 == dim:
        raise ValueError('dimensions do not match')

    corner_keys = coords_to_keys(T, extent.corners)

    shape = np.ptp(corner_keys, axis=0) + 1

    # Minimum corner
    idx = np.argmin(corner_keys, axis=0)
    origin_key = corner_keys[idx, range(dim)]
    min_corner = keys_to_coords(T, [origin_key - 0.5])[0, :]

    # define transformation
    t = np.copy(T)
    t[:-1, dim] = min_corner

    return t, origin_key, shape
