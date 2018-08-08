# BEGIN OF LICENSE NOTE
# This file is part of PoYnts.
# Copyright (c) 2018, Sebastian Lamprecht, lamprecht@uni-trier.de
# 
# This software is copyright protected. A decision on a less restrictive
# licencing model will be made before releasing this software.
# END OF LICENSE NOTE
"""Handling of grided data, like voxels or rasters.
"""

import numpy as np

from .. import (
    assertion,
    nptools,
    projection,
    GeoRecords,
    Extent
)

from . import (
    convert
)

from .transformation import (
    keys_to_coords,
    coords_to_keys,
    coords_to_coords,
    keys_to_indices,
    indices_to_keys,
)


class Grid(GeoRecords):
    """Grid class extends GeoRecords to ease handling of matrices, like rasters
    or voxels.

    Parameters
    ----------
    proj : pointspy.projection.Proj
        Projection object provides the geograpic projection of the grid.
    rec : np.recarray
        Multidimensional array of objects. Element of the matrix represents a
        object with k coordinate dimension.
    T : array_like(Number, shape=(k+1, k+1))
        The  linear transformation matrix to transform the coordinates.
        The translation represents the origin, the rotation the
        orientation and the scale the pixel size of the matrix.

    Examples
    --------

    Create a raster with projection and a transformation matrix.

    >>> from pointspy import transformation
    >>> proj = projection.Proj()
    >>> data = np.recarray((4, 3), dtype=[('values', int)])
    >>> data['values'] = np.arange(np.product(data.shape)).reshape(data.shape)
    >>> T = transformation.matrix(t=[10, 20], s=[0.5, 0.4], order='rst')

    >>> raster = Grid(proj, data, T)
    >>> print(raster.dtype.descr)
    [('values', '<i8'), ('coords', '<f8', (2,))]
    >>> print(raster.t.origin)
    [10. 20.]
    >>> print(raster.shape)
    (4, 3)
    >>> print(raster.dim)
    2

    >>> print(raster.coords)
    [[[10.25 20.2 ]
      [10.75 20.2 ]
      [11.25 20.2 ]]
    <BLANKLINE>
     [[10.25 20.6 ]
      [10.75 20.6 ]
      [11.25 20.6 ]]
    <BLANKLINE>
     [[10.25 21.  ]
      [10.75 21.  ]
      [11.25 21.  ]]
    <BLANKLINE>
     [[10.25 21.4 ]
      [10.75 21.4 ]
      [11.25 21.4 ]]]
    >>> print(raster.values)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]

    Convert coordinates to indices and reverse.

    >>> print(raster.coords_to_keys(raster.t.origin))
    [0 0]
    >>> print(raster.keys_to_coords([-0.5, -0.5]))
    [10. 20.]

    >>> print(raster.coords_to_keys(raster.coords))
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
    >>> print(raster.keys_to_coords(raster.keys))
    [[[10.25 20.2 ]
      [10.75 20.2 ]
      [11.25 20.2 ]]
    <BLANKLINE>
     [[10.25 20.6 ]
      [10.75 20.6 ]
      [11.25 20.6 ]]
    <BLANKLINE>
     [[10.25 21.  ]
      [10.75 21.  ]
      [11.25 21.  ]]
    <BLANKLINE>
     [[10.25 21.4 ]
      [10.75 21.4 ]
      [11.25 21.4 ]]]

    Use spatial index.

    >>> dists, indices = raster.indexKD().knn(raster.t.origin, k=5)
    >>> print(dists)
    [0.32015621 0.65       0.77620873 0.96046864 1.03077641]
    >>> print(indices)
    [0 3 1 4 6]
    >>> print(raster.indices_to_keys(indices))
    [[0 0]
     [1 0]
     [0 1]
     [1 1]
     [2 0]]

    See Also
    --------
    GeoRecords

    """
    def __new__(cls, proj, rec, T):

        if not isinstance(proj, projection.Proj):
            raise TypeError("'proj' needs to be of type 'Proj'")
        if not isinstance(rec, np.recarray):
            raise TypeError('numpy record array required')
        if not len(rec.shape) > 1:
            raise ValueError('at least two dimensions required')
        T = assertion.ensure_tmatrix(T)

        if 'coords' not in rec.dtype.names:
            keys = nptools.indices(rec.shape)
            coords = keys_to_coords(T, keys)
            dtype = [('coords', float, len(rec.shape))]
            rec = nptools.add_fields(rec, dtype, data=[coords])

        grid = GeoRecords(proj, rec, T=T).reshape(rec.shape).view(cls)
        return grid

    def transform(self, T):
        """Transform coordinates.

        Parameters
        ----------
        T : array_like(Number, shape=(self.dim+1, self.dim+1))
            Transformation matrix to apply.

        Notes
        -----
        Overwrites `GeoRecords.transform`.

        See Also
        --------
        GeoRecords.transform

        """
        T = assertion.ensure_tmatrix(T, min_dim=self.dim, max_dim=self.dim)
        self.t = T * self.t
        self.coords[:] = self.keys_to_coords(self.keys)

    def keys_to_indices(self, keys):
        """Convert grid keys to indices.

        See Also
        --------
        poynts.grid.keys_to_indices

        """
        return keys_to_indices(keys, self.shape)

    def indices_to_keys(self, indices):
        """Convert grid indices to keys.

        See Also
        --------
        poynts.grid.indices_to_keys

        """
        return indices_to_keys(indices, self.shape)

    def keys_to_coords(self, keys):
        """Convert raster indices to coordinates.

        See Also
        --------
        poynts.grid.keys_to_coords

        """
        return keys_to_coords(self.t, keys)

    def coords_to_keys(self, coords):
        """Convert coordinates to raster indices.

        See Also
        --------
        poynts.grid.coords_to_keys

        """
        return coords_to_keys(self.t, coords)

    def coords_to_coords(self, coords):
        """Set coordinate to closest grid coordinate.

        See Also
        --------
        poynts.grid.coords_to_coords

        """
        return coords_to_coords(self.t, coords)

    def window_by_extent(self, extent):
        """Get grid subset within extent.

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

        >>> from pointspy import transformation

        Create Grid.

        >>> T = transformation.matrix(s=(1, 2), t=[1, 0])
        >>> proj = projection.Proj()
        >>> grid = Grid(proj, np.recarray((4, 5), dtype=[]), T=T)
        >>> print(grid.extent())
        [1.5 1.  5.5 7. ]

        Select subset by extent.

        >>> subset = grid.window_by_extent([3, 2, 7, 6])
        >>> print(subset.shape)
        (2, 3)
        >>> print(subset.coords)
        [[[3.5 3. ]
          [4.5 3. ]
          [5.5 3. ]]
        <BLANKLINE>
         [[3.5 5. ]
          [4.5 5. ]
          [5.5 5. ]]]

        """
        extent = Extent(extent)
        coords = self.coords_to_keys(extent.corners)
        min_idx = coords.min(0)
        max_idx = coords.max(0)
        slices = [slice(min_idx[i], max_idx[i], 1) for i in range(self.dim)]
        return self[slices]

    def voxelize(self, rec, **kwargs):
        """Convert a point cloud to a voxel or raster.

        Parameters
        ----------
        rec : np.recarray(shape=(n, ))
            Numpy record array of `n` points to voxelize. It reqires the two
            dimensional field 'coords' associated with `k` dimensional
            coordinates.
        \*\*kwargs
            Arguments passed to voxelize.

        See Also
        --------
        voxelize

        Examples
        --------

        >>> from pointspy import transformation

        Create Grid.

        >>> T = transformation.matrix(s=(2.5, 3), t=[1, 0])
        >>> dtype = [('points', np.recarray)]
        >>> proj = projection.Proj()
        >>> grid = Grid(proj, np.recarray((3, 4), dtype=dtype), T=T)

        Create records to voxelize.

        >>> coords = [(0, 0), (1, 0.5), (2, 2), (4, 6), (3, 2), (1, 5), (3, 5)]
        >>> rec = np.recarray(len(coords), dtype=[('coords', float, 2)])
        >>> rec['coords'] = coords

        Voxelize and update grid.

        >>> voxels = grid.voxelize(rec)
        >>> grid['points'] = voxels

        >>> print(grid['points'][0, 0].coords)
        [[1.  0.5]
         [2.  2. ]
         [3.  2. ]]

        """
        return convert.voxelize(rec, self.t, shape=self.shape, **kwargs)
