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
"""Handling of grided data like voxels or rasters.
"""

import numpy as np
import pandas

from .. import (
    assertion,
    nptools,
    projection,
)
from .. georecords import GeoRecords
from .. extent import Extent
from .transformation import (
    keys_to_coords,
    coords_to_keys,
    coords_to_coords,
    keys_to_indices,
    indices_to_keys,
    corners_to_transform,
    transform_to_corners,
)
from ..misc import print_rounded


class Grid(GeoRecords):
    """Grid class extends GeoRecords to ease handling of matrices like rasters
    or voxels.

    Parameters
    ----------
    proj : pyoints.projection.Proj
        Projection object provides the geograpic projection of the grid.
    rec : np.recarray
        Multidimensional array of objects. Each cell of the matrix represents a
        geo-object with `k` dimensional coordinates.
    T : array_like(Number, shape=(k+1, k+1))
        A linear transformation matrix to transform the coordinates. The
        translation represents the origin, the rotation represents the
        orientation, and the scale represents the pixel size of the matrix.

    Examples
    --------

    >>> from pyoints import (
    ...     transformation,
    ...     projection,
    ... )

    Create a raster with a projection and a transformation matrix.

    >>> proj = projection.Proj()
    >>> data = np.recarray((4, 3), dtype=[('values', int)])
    >>> data['values'] = np.arange(np.product(data.shape)).reshape(data.shape)
    >>> T = transformation.matrix(t=[10, 20], s=[0.5, 0.4], order='rst')

    >>> raster = Grid(proj, data, T)
    >>> print(raster.dtype.descr)
    [('values', '<i8'), ('coords', '<f8', (2,))]
    >>> print_rounded(raster.shape)
    (4, 3)
    >>> print_rounded(raster.dim)
    2
    >>> print_rounded(raster.t.origin)
    [ 10.  20.]

    Get cell data of the raster.

    >>> print_rounded(raster.coords)
    [[[ 10.25  20.2 ]
      [ 10.75  20.2 ]
      [ 11.25  20.2 ]]
    <BLANKLINE>
     [[ 10.25  20.6 ]
      [ 10.75  20.6 ]
      [ 11.25  20.6 ]]
    <BLANKLINE>
     [[ 10.25  21.  ]
      [ 10.75  21.  ]
      [ 11.25  21.  ]]
    <BLANKLINE>
     [[ 10.25  21.4 ]
      [ 10.75  21.4 ]
      [ 11.25  21.4 ]]]
    >>> print_rounded(raster.values)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]

    Convert coordinates to indices and reverse.

    >>> print_rounded(raster.coords_to_keys(raster.t.origin))
    [0 0]
    >>> print_rounded(raster.keys_to_coords([-0.5, -0.5]))
    [ 10.  20.]

    >>> print_rounded(raster.coords_to_keys(raster.coords))
    [[[0 0]
      [0 1]
      [0 2]]
    <BLANKLINE>
     [[1 0]
      [1 1]
      [1 2]]
    <BLANKLINE>
     [[2 0]
      [2 1]
      [2 2]]
    <BLANKLINE>
     [[3 0]
      [3 1]
      [3 2]]]
    >>> print_rounded(raster.keys_to_coords(raster.keys))
    [[[ 10.25  20.2 ]
      [ 10.75  20.2 ]
      [ 11.25  20.2 ]]
    <BLANKLINE>
     [[ 10.25  20.6 ]
      [ 10.75  20.6 ]
      [ 11.25  20.6 ]]
    <BLANKLINE>
     [[ 10.25  21.  ]
      [ 10.75  21.  ]
      [ 11.25  21.  ]]
    <BLANKLINE>
     [[ 10.25  21.4 ]
      [ 10.75  21.4 ]
      [ 11.25  21.4 ]]]

    Usage of the spatial index.

    >>> dists, indices = raster.indexKD().knn(raster.t.origin, k=5)
    >>> print_rounded(dists)
    [ 0.32  0.65  0.78  0.96  1.03]
    >>> print_rounded(indices)
    [0 3 1 4 6]
    >>> print_rounded(raster.indices_to_keys(indices))
    [[0 0]
     [1 0]
     [0 1]
     [1 1]
     [2 0]]

    See Also
    --------
    GeoRecords

    """
    def __new__(cls, proj, rec, T, date=None):

        if not isinstance(proj, projection.Proj):
            m = "'proj' needs to be of type 'Proj', got %s" % type(proj)
            raise TypeError(m)
        if not isinstance(rec, np.recarray):
            raise TypeError('numpy record array required, got %s' % type(rec))
        if not len(rec.shape) > 1:
            raise ValueError('at least two dimensions required')
        T = assertion.ensure_tmatrix(T)

        if 'coords' not in rec.dtype.names:
            keys = nptools.indices(rec.shape)
            coords = keys_to_coords(T, keys)
            dtype = [('coords', float, len(rec.shape))]
            rec = nptools.add_fields(rec, dtype, data=[coords])

        grid = GeoRecords(proj, rec, T=T, date=date)
        return grid.reshape(rec.shape).view(cls)

    @classmethod
    def from_corners(cls, proj, corners, scale):
        """Alternative constructor using grid corners.

        Parameters
        ----------
        proj : pyoints.projection.Proj
            Projection object provides the geographic projection of the grid.
        corners : array_like(Number, shape=(n, k))
            Desired `k` dimensional corners of the grid.
        scale : array_like(int, shape=(k))
            Desired scale of the pixel cells.

        See also
        --------
        Grid.from_extent

        Examples
        --------

        Derive a raster form corners with a positive scale.

        >>> corners = [(1, 2), (4, 2), (4, 10), (1, 10)]
        >>> scale = [1, 2]
        >>> raster = Grid.from_corners(projection.Proj(), corners, scale)

        >>> print_rounded(raster.shape)
        (4, 3)
        >>> print(sorted(raster.dtype.descr))
        [('coords', '<f8', (2,))]
        >>> print_rounded(raster.t)
        [[ 1.  0.  1.]
         [ 0.  2.  2.]
         [ 0.  0.  1.]]

        >>> print_rounded(raster.coords)
        [[[ 1.5  3. ]
          [ 2.5  3. ]
          [ 3.5  3. ]]
        <BLANKLINE>
         [[ 1.5  5. ]
          [ 2.5  5. ]
          [ 3.5  5. ]]
        <BLANKLINE>
         [[ 1.5  7. ]
          [ 2.5  7. ]
          [ 3.5  7. ]]
        <BLANKLINE>
         [[ 1.5  9. ]
          [ 2.5  9. ]
          [ 3.5  9. ]]]

        >>> print_rounded(raster.corners)
        [[  1.   2.]
         [  4.   2.]
         [  4.  10.]
         [  1.  10.]]
        >>> keys = raster.t.to_global(corners)
        >>> print_rounded(keys)
        [[ 0.  0.]
         [ 3.  0.]
         [ 3.  4.]
         [ 0.  4.]]
        >>> print_rounded(raster.t.to_local(keys))
        [[  1.   2.]
         [  4.   2.]
         [  4.  10.]
         [  1.  10.]]

        Example with inverted axes.

        >>> from pyoints import transformation

        >>> corners = [(1, 1), (3, 1), (3, 4), (1, 4)]
        >>> raster = Grid.from_corners(projection.Proj(), corners, [-0.5, -1])

        >>> print_rounded(raster.shape)
        (3, 4)
        >>> print_rounded(raster.records().coords.min(0))
        [ 1.25  1.5 ]
        >>> print_rounded(raster.records().coords.max(0))
        [ 2.75  3.5 ]

        >>> t, r, s, det = transformation.decomposition(raster.t)
        >>> print_rounded(t)
        [ 3.  4.]
        >>> print_rounded(s)
        [ 0.5  1. ]

        >>> print_rounded(raster.coords)
        [[[ 2.75  3.5 ]
          [ 2.25  3.5 ]
          [ 1.75  3.5 ]
          [ 1.25  3.5 ]]
        <BLANKLINE>
         [[ 2.75  2.5 ]
          [ 2.25  2.5 ]
          [ 1.75  2.5 ]
          [ 1.25  2.5 ]]
        <BLANKLINE>
         [[ 2.75  1.5 ]
          [ 2.25  1.5 ]
          [ 1.75  1.5 ]
          [ 1.25  1.5 ]]]

        >>> print_rounded(raster.corners)
        [[ 3.  4.]
         [ 1.  4.]
         [ 1.  1.]
         [ 3.  1.]]
        >>> keys = raster.t.to_global(corners)
        >>> print_rounded(keys)
        [[ 4.  3.]
         [ 0.  3.]
         [ 0.  0.]
         [ 4.  0.]]
        >>> print_rounded(raster.t.to_local(keys))
        [[ 1.  1.]
         [ 3.  1.]
         [ 3.  4.]
         [ 1.  4.]]

        """
        T, shape = corners_to_transform(corners, scale=scale)
        keys = nptools.indices(shape)
        coords = keys_to_coords(T, keys)
        rec = np.recarray(shape, dtype=[('coords', float, len(shape))])
        rec['coords'] = coords
        return cls(proj, rec, T)

    @classmethod
    def from_extent(cls, proj, ext, scale):
        """Alternative constructor using grid corners.

        Parameters
        ----------
        proj : pyoints.projection.Proj
            Projection object provides the geographic projection of the grid.
        ext : array_like(Number, shape=(2 * k)) or array_like(Number, shape=(n, k))
            Desired `k` dimensional extent of the gird. You can also specify
            the extent by providing coordinates.
        scale : array_like(int, shape=(k))
            Desired scale of the pixel cells.

        See also
        --------
        Grid.from_corners

        Examples
        --------

        Defining a raster using a two dimensional extent.

        >>> extent = [1, 1, 3, 4]
        >>> raster = Grid.from_extent(projection.Proj(), extent, [0.5, 1])

        >>> print_rounded(raster.shape)
        (3, 4)
        >>> print(sorted(raster.dtype.descr))
        [('coords', '<f8', (2,))]
        >>> print_rounded(raster.coords)
        [[[ 1.25  1.5 ]
          [ 1.75  1.5 ]
          [ 2.25  1.5 ]
          [ 2.75  1.5 ]]
        <BLANKLINE>
         [[ 1.25  2.5 ]
          [ 1.75  2.5 ]
          [ 2.25  2.5 ]
          [ 2.75  2.5 ]]
        <BLANKLINE>
         [[ 1.25  3.5 ]
          [ 1.75  3.5 ]
          [ 2.25  3.5 ]
          [ 2.75  3.5 ]]]

        """
        corners = Extent(ext).corners
        return cls.from_corners(proj, corners, scale)

    @property
    def corners(self):
        return transform_to_corners(self.t, self.shape)

    def transform(self, T):
        """Transforms coordinates.

        Parameters
        ----------
        T : array_like(Number, shape=(self.dim+1, self.dim+1))
            Transformation matrix to apply.

        Returns
        -------
        self

        Notes
        -----
        Overwrites `GeoRecords.transform`.

        See Also
        --------
        GeoRecords.transform

        """
        T = assertion.ensure_tmatrix(T, dim=self.dim)
        self.t = T @ self.t
        self['coords'] = self.keys_to_coords(self.keys)
        self._clear_cache()
        return self

    def keys_to_indices(self, keys):
        """Converts grid keys to indices.

        See Also
        --------
        poynts.grid.keys_to_indices

        """
        return keys_to_indices(keys, self.shape)

    def indices_to_keys(self, indices):
        """Converts cell indices to keys.

        See Also
        --------
        poynts.grid.indices_to_keys

        """
        return indices_to_keys(indices, self.shape)

    def keys_to_coords(self, keys):
        """Converts cell indices to coordinates.

        See Also
        --------
        poynts.grid.keys_to_coords

        """
        return keys_to_coords(self.t, keys)

    def coords_to_keys(self, coords):
        """Converts coordinates to cell indices.

        See Also
        --------
        poynts.grid.coords_to_keys

        """
        return coords_to_keys(self.t, coords)

    def coords_to_coords(self, coords):
        """Sets coordinate to closest grid coordinate.

        See Also
        --------
        poynts.grid.coords_to_coords

        """
        return coords_to_coords(self.t, coords)

    def window_by_extent(self, extent):
        """Gets a subset of the grid within a given extent.

        Parameters
        ----------
        extent : Extent
            Extent defining the window corners.

        Returns
        -------
        Grid
            Desired subset.

        Examples
        --------

        >>> from pyoints import transformation

        Create Grid.

        >>> T = transformation.matrix(s=(1, 2), t=[1, 0])
        >>> proj = projection.Proj()
        >>> grid = Grid(proj, np.recarray((4, 5), dtype=[]), T=T)
        >>> print_rounded(grid.extent())
        [ 1.5  1.   5.5  7. ]

        Select subset by extent.

        >>> subset = grid.window_by_extent([3, 2, 7, 6])
        >>> print_rounded(subset.shape)
        (2, 3)
        >>> print_rounded(subset.coords)
        [[[ 3.5  3. ]
          [ 4.5  3. ]
          [ 5.5  3. ]]
        <BLANKLINE>
         [[ 3.5  5. ]
          [ 4.5  5. ]
          [ 5.5  5. ]]]

        """
        extent = Extent(extent)
        coords = self.coords_to_keys(extent.corners)
        min_idx = coords.min(0)
        max_idx = coords.max(0)
        slices = [slice(min_idx[i], max_idx[i], 1) for i in range(self.dim)]

        return self[tuple(slices)]

    def voxelize(self, rec, **kwargs):
        """Converts a point cloud to a voxel or raster.

        Parameters
        ----------
        rec : np.recarray(shape=(n, ))
            Numpy record array of `n` points to voxelize. It requires the two
            dimensional field 'coords' associated with `k` dimensional
            coordinates.
        \*\*kwargs
            Arguments passed to voxelize.

        See Also
        --------
        voxelize

        Examples
        --------

        >>> from pyoints import (
        ...     transformation,
        ...     projection,
        ... )

        Create Grid.

        >>> T = transformation.matrix(s=(2.5, 3), t=[1, 0])
        >>> dtype = [('points', np.recarray)]
        >>> proj = projection.Proj()
        >>> grid = Grid(proj, np.recarray((3, 4), dtype=dtype), T=T)

        >>> print_rounded(grid.shape)
        (3, 4)
        >>> print_rounded(grid.coords)
        [[[ 2.25  1.5 ]
          [ 4.75  1.5 ]
          [ 7.25  1.5 ]
          [ 9.75  1.5 ]]
        <BLANKLINE>
         [[ 2.25  4.5 ]
          [ 4.75  4.5 ]
          [ 7.25  4.5 ]
          [ 9.75  4.5 ]]
        <BLANKLINE>
         [[ 2.25  7.5 ]
          [ 4.75  7.5 ]
          [ 7.25  7.5 ]
          [ 9.75  7.5 ]]]
        >>> print_rounded(grid.t.origin)
        [ 1.  0.]

        Create records to voxelize.

        >>> coords = [(0, 0), (1, 0.5), (2, 2), (4, 6), (3, 2), (1, 5), (3, 5)]
        >>> rec = np.recarray(len(coords), dtype=[('coords', float, 2)])
        >>> rec['coords'] = coords

        Voxelize the records and update the raster.

        >>> voxels = grid.voxelize(rec)
        >>> grid['points'] = voxels

        >>> print_rounded(grid['points'][0, 0].coords)
        [[ 0.   0. ]
         [ 1.   0.5]
         [ 2.   2. ]
         [ 3.   2. ]]

        """
        return voxelize(rec, self.t, shape=self.shape, **kwargs)


def voxelize(rec, T, shape=None, agg_func=None, dtype=None):
    """Aggregates a point cloud to a voxel or raster.

    Parameters
    ----------
    rec : np.recarray(shape=(n, )) or GeoRecords
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
        Output data type. If None, set to automatically.

    Returns
    -------
    np.ndarray or np.recarray(dtype=dtype) or Grid
        Desired `m` dimensional matrix. If `rec` is an instance of `GeoRecords`
        and `dtype` has named fields, an instance of `Grid` is returned. If
        no point falls within `T`, None is returned.

    See Also
    --------
    Grid, np.apply_function

    Examples
    --------

    >>> from pyoints import transformation

    Create record array with coordinates.

    >>> coords = [(0, 0), (1, 0.5), (2, 2), (4, 6), (3, 2), (1, 5), (3, 0)]
    >>> rec = np.recarray(len(coords), dtype=[('coords', float, 2)])
    >>> rec['coords'] = coords

    Voxelize records.

    >>> T = transformation.matrix(s=(2.5, 3), t=[0, 0])
    >>> grid = voxelize(rec, T)

    >>> print(grid.dtype)
    object
    >>> print_rounded(grid.shape)
    (3, 2)
    >>> print_rounded(grid[0, 0].coords)
    [[ 0.   0. ]
     [ 1.   0.5]
     [ 2.   2. ]]

    Voxelize with aggregation function.

    >>> dtype = [('points', type(rec)), ('count', int)]
    >>> agg_func = lambda ids: (rec[ids], len(ids))
    >>> grid = voxelize(rec, T, agg_func=agg_func, dtype=dtype)

    >>> print(grid.dtype.descr)
    [('points', '|O'), ('count', '<i8')]
    >>> print_rounded(grid.shape)
    (3, 2)
    >>> print_rounded(grid.count)
    [[3 2]
     [1 0]
     [0 1]]
    >>> print_rounded(grid[0, 0].points.coords)
    [[ 0.   0. ]
     [ 1.   0.5]
     [ 2.   2. ]]

    Voxelize three dimensional coordinates to receive a two dimensional raster.

    >>> coords = [(0, 0, 1), (-2, 0.3, 5), (2, 2, 3), (4, 6, 2), (3, 2, 1)]
    >>> rec = np.recarray(len(coords), dtype=[('coords', float, 3)])
    >>> rec['coords'] = coords

    >>> T = transformation.matrix(s=(2.5, 3), t=[0, 0])
    >>> grid = voxelize(rec, T)
    >>> print_rounded(grid.shape)
    (3, 2)
    >>> print_rounded(grid[0, 0].coords)
    [[ 0.   0.   1. ]
     [-2.   0.3  5. ]
     [ 2.   2.   3. ]]

    """
    if not isinstance(rec, np.recarray):
        raise TypeError("'rec' needs an instance of np.recarray")
    if 'coords' not in rec.dtype.names:
        raise ValueError("'rec' requires field 'coords'")

    if agg_func is None:
        def agg_func(ids): return rec[ids]
    elif not callable(agg_func):
        raise ValueError("'agg_func' needs to be callable")

    coords = assertion.ensure_coords(rec.coords, min_dim=T.dim)
    keys = coords_to_keys(T, coords[:, :T.dim])

    if shape is None:
        shape = keys.max(0)[:T.dim] + 1
    else:
        shape = assertion.ensure_shape(shape, dim=T.dim)
    if np.any(np.array(shape) < 0):
        return None

    # group keys
    df = pandas.DataFrame({'indices': keys_to_indices(keys, shape)})
    groupDict = df.groupby(by=df.indices).groups
    keys = indices_to_keys(list(groupDict.keys()), shape)

    # cut by extent
    min_mask = np.all(keys >= 0, axis=1)
    max_mask = np.all(keys < shape, axis=1)
    mask = np.where(np.all((min_mask, max_mask), axis=0))[0]

    # create lookup array
    lookup = np.empty(shape, dtype=list)
    lookup.fill([])

    values = list(groupDict.values())
    lookup[tuple(keys[mask].T.tolist())] = [values[i] for i in mask]

    # Aggregate per cell
    try:
        res = nptools.apply_function(lookup, agg_func, dtype=dtype)
    except BaseException:
        m = "aggregation failed, please check 'agg_func' and 'dtype'"
        raise ValueError(m)

    if isinstance(rec, GeoRecords) and isinstance(res, np.recarray):
        res = Grid(rec.proj, res, T, date=rec.date)

    return res
